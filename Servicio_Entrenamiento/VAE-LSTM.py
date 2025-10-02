import os
import glob
import argparse
import pickle
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler


class SequenceDataset(Dataset):

    def __init__(
        self,
        csv_paths: List[str],
        sequence_length: int,
        scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ) -> None:
        self.sequence_length = sequence_length
        self.scaler = scaler or StandardScaler()

        data_frames: List[pd.DataFrame] = []
        for path in csv_paths:
            df = pd.read_csv(path)
            # Seleccionar solo coordenadas x, y, z
            feature_cols = [
                c for c in df.columns
                if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")
            ]
            if len(feature_cols) == 0:
                continue
            data_frames.append(df[feature_cols])

        if len(data_frames) == 0:
            raise ValueError("No se encontraron columnas de características (_x,_y,_z) en los CSV proporcionados.")

        # Concatenar a lo largo del tiempo (uno tras otro)
        full_df = pd.concat(data_frames, axis=0, ignore_index=True)
        features = full_df.values.astype(np.float32)

        if fit_scaler:
            self.scaler.fit(features)

        features = self.scaler.transform(features)

        # Crear secuencias deslizantes
        self.num_features = features.shape[1]
        self.sequences: List[np.ndarray] = []
        for start in range(0, len(features) - sequence_length + 1):
            window = features[start:start + sequence_length]
            self.sequences.append(window)

        if len(self.sequences) == 0:
            raise ValueError("No se pudieron crear secuencias. ¿El CSV es demasiado corto para la ventana especificada?")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq = self.sequences[idx]
        return torch.from_numpy(seq)  # shape: [seq_len, num_features]


class VAELSTM(nn.Module):
    """VAE con encoder LSTM y decoder LSTM simple.
    Encoder: LSTM -> estado oculto final -> mu, logvar
    Reparametrización: z = mu + eps * std
    Decoder: alimenta z repetido como input en cada paso -> LSTM -> Linear -> features
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int = 256,
        latent_dim: int = 64,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_size, num_features)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, T, F]
        _, (h_n, _) = self.encoder_lstm(x)
        h_last = h_n[-1]  # [B, H]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        # z: [B, Z] -> repeat across time as decoder input
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)  # [B, T, Z]
        out, _ = self.decoder_lstm(z_seq)
        recon = self.output_fc(out)  # [B, T, F]
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, x.size(1))
        return recon, mu, logvar


def loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction="mean")
    # KL(N(mu, sigma) || N(0,1))
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl


def discover_processed_csvs(project_root: str) -> List[str]:
    coord_dir = os.path.join(project_root, "Coordenadas_csv")
    paths = sorted(glob.glob(os.path.join(coord_dir, "*_processed.csv")))
    if len(paths) == 0:
        raise FileNotFoundError(f"No se encontraron CSV procesados en: {coord_dir}")
    return paths


def train(
    project_root: str,
    sequence_length: int = 60,
    batch_size: int = 32,
    epochs: int = 30,
    hidden_size: int = 256,
    latent_dim: int = 64,
    lr: float = 1e-3,
    beta: float = 1.0,
    val_split: float = 0.1,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_paths = discover_processed_csvs(project_root)

    # Dataset para fit de scaler
    full_dataset_for_scaler = SequenceDataset(
        csv_paths=csv_paths,
        sequence_length=sequence_length,
        scaler=None,
        fit_scaler=True,
    )

    scaler = full_dataset_for_scaler.scaler

    # Dataset definitivo con scaler ya entrenado
    dataset = SequenceDataset(
        csv_paths=csv_paths,
        sequence_length=sequence_length,
        scaler=scaler,
        fit_scaler=False,
    )

    num_features = full_dataset_for_scaler.num_features

    # Split train/val
    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = VAELSTM(
        num_features=num_features,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        num_layers=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    save_dir = os.path.join(project_root, "Servicio_IA_Entrenada")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(batch)
            loss = loss_function(recon, batch, mu, logvar, beta=beta)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch)
                loss = loss_function(recon, batch, mu, logvar, beta=beta)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if len(train_losses) else 0.0
        val_loss = float(np.mean(val_losses)) if len(val_losses) else 0.0
        print(f"Epoch {epoch:03d} | train: {train_loss:.6f} | val: {val_loss:.6f}")

        # Guardado del mejor modelo
        if val_loss < best_val:
            best_val = val_loss
            model_path = os.path.join(save_dir, "vae_lstm_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_features": num_features,
                "hidden_size": hidden_size,
                "latent_dim": latent_dim,
                "sequence_length": sequence_length,
                "scaler": scaler,  # también guardamos el scaler (pickle-able)
            }, model_path)
            print(f"Guardado mejor modelo en: {model_path}")

    # Guardar último modelo y scaler explícitamente
    last_model_path = os.path.join(save_dir, "vae_lstm_last.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_features": num_features,
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
        "sequence_length": sequence_length,
    }, last_model_path)

    scaler_path = os.path.join(save_dir, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("\nEntrenamiento finalizado.")
    print(f"Modelo (último): {last_model_path}")
    print(f"Scaler: {scaler_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar VAE-LSTM sobre coordenadas procesadas")
    parser.add_argument("--sequence_length", type=int, default=60, help="Longitud de la ventana temporal")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--epochs", type=int, default=30, help="Épocas de entrenamiento")
    parser.add_argument("--hidden_size", type=int, default=256, help="Tamaño oculto LSTM")
    parser.add_argument("--latent_dim", type=int, default=64, help="Dimensión latente VAE")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Peso KL en la pérdida")
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    train(
        project_root=project_root,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        lr=args.lr,
        beta=args.beta,
    )