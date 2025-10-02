import os
import glob
import argparse
import pickle
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class StepDataset(Dataset):
    def __init__(
        self,
        csv_paths: List[str],
        max_length: int,
        scaler: StandardScaler | None = None,
        label_encoder: LabelEncoder | None = None,
        fit_scaler: bool = False,
        fit_encoder: bool = False,
    ) -> None:
        self.max_length = max_length
        self.scaler = scaler or StandardScaler()
        self.label_encoder = label_encoder or LabelEncoder()

        self.sequences: List[np.ndarray] = []
        self.labels: List[str] = []
        self.lengths: List[int] = []

        all_features = []
        dance_labels = []

        for path in csv_paths:
            df = pd.read_csv(path)
            feature_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")]
            if len(feature_cols) == 0:
                continue

            features = df[feature_cols].values.astype(np.float32)
            dance_name = os.path.basename(path).split('_')[0]
            
            all_features.append(features)
            dance_labels.append(dance_name)

        if len(all_features) == 0:
            raise ValueError("No se encontraron características en los CSV.")

        if fit_scaler:
            concat_features = np.vstack(all_features)
            self.scaler.fit(concat_features)

        if fit_encoder:
            self.label_encoder.fit(dance_labels)

        for features, dance_name in zip(all_features, dance_labels):
            features_scaled = self.scaler.transform(features)
            original_len = len(features_scaled)
            
            if original_len > max_length:
                features_scaled = features_scaled[:max_length]
                actual_len = max_length
            else:
                padding = np.zeros((max_length - original_len, features_scaled.shape[1]), dtype=np.float32)
                features_scaled = np.vstack([features_scaled, padding])
                actual_len = original_len

            self.sequences.append(features_scaled)
            self.labels.append(dance_name)
            self.lengths.append(actual_len)

        self.num_features = self.sequences[0].shape[1]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = torch.from_numpy(self.sequences[idx])
        label = torch.tensor(self.label_encoder.transform([self.labels[idx]])[0], dtype=torch.long)
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        return seq, label, length


class CVAELSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_classes: int,
        hidden_size: int = 256,
        latent_dim: int = 64,
        num_layers: int = 1,
        embedding_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        
        self.encoder_lstm = nn.LSTM(
            input_size=num_features + embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim + embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_size, num_features)

    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        class_emb = self.class_embedding(labels).unsqueeze(1).repeat(1, x.size(1), 1)
        x_cond = torch.cat([x, class_emb], dim=-1)
        _, (h_n, _) = self.encoder_lstm(x_cond)
        h_last = h_n[-1]
        mu = self.fc_mu(h_last)
        logvar = self.fc_logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor, seq_len: int) -> torch.Tensor:
        class_emb = self.class_embedding(labels).unsqueeze(1).repeat(1, seq_len, 1)
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)
        z_cond = torch.cat([z_seq, class_emb], dim=-1)
        out, _ = self.decoder_lstm(z_cond)
        recon = self.output_fc(out)
        return recon

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels, x.size(1))
        return recon, mu, logvar


def loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, 
                  lengths: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    batch_size = x.size(0)
    max_len = x.size(1)
    
    mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()
    
    recon_loss = torch.sum(((recon_x - x) ** 2) * mask) / torch.sum(mask)
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
    max_length: int = 150,
    batch_size: int = 2,
    epochs: int = 50,
    hidden_size: int = 256,
    latent_dim: int = 64,
    embedding_dim: int = 32,
    lr: float = 1e-3,
    beta: float = 1.0,
    val_split: float = 0.5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_paths = discover_processed_csvs(project_root)
    print(f"Encontrados {len(csv_paths)} archivos CSV")

    full_dataset_for_scaler = StepDataset(
        csv_paths=csv_paths,
        max_length=max_length,
        scaler=None,
        label_encoder=None,
        fit_scaler=True,
        fit_encoder=True,
    )

    scaler = full_dataset_for_scaler.scaler
    label_encoder = full_dataset_for_scaler.label_encoder
    num_classes = len(label_encoder.classes_)
    
    print(f"Danzas encontradas: {label_encoder.classes_}")

    dataset = StepDataset(
        csv_paths=csv_paths,
        max_length=max_length,
        scaler=scaler,
        label_encoder=label_encoder,
        fit_scaler=False,
        fit_encoder=False,
    )

    num_features = dataset.num_features

    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    print(f"Dataset total: {len(dataset)}, Train: {train_len}, Val: {val_len}")

    # Ajustar batch_size si es necesario
    effective_batch_size = min(batch_size, train_len)
    print(f"Batch size ajustado: {effective_batch_size}")

    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, val_len), shuffle=False, drop_last=False)

    model = CVAELSTM(
        num_features=num_features,
        num_classes=num_classes,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        num_layers=1,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    save_dir = os.path.join(project_root, "Servicio_IA_Entrenada")
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        train_batch_count = 0
        for seq, labels, lengths in train_loader:
            seq, labels, lengths = seq.to(device), labels.to(device), lengths.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(seq, labels)
            loss = loss_function(recon, seq, mu, logvar, lengths, beta=beta)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            train_batch_count += 1
        
        if epoch == 1:
            print(f"Epoch 1 - Train batches procesados: {train_batch_count}")

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for seq, labels, lengths in val_loader:
                seq, labels, lengths = seq.to(device), labels.to(device), lengths.to(device)
                recon, mu, logvar = model(seq, labels)
                loss = loss_function(recon, seq, mu, logvar, lengths, beta=beta)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if len(train_losses) else 0.0
        val_loss = float(np.mean(val_losses)) if len(val_losses) else 0.0
        print(f"Epoch {epoch:03d} | train: {train_loss:.6f} | val: {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            model_path = os.path.join(save_dir, "cvae_lstm_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_features": num_features,
                "num_classes": num_classes,
                "hidden_size": hidden_size,
                "latent_dim": latent_dim,
                "embedding_dim": embedding_dim,
                "max_length": max_length,
                "scaler": scaler,
                "label_encoder": label_encoder,
            }, model_path)
            print(f"Guardado mejor modelo: {model_path}")

    last_model_path = os.path.join(save_dir, "cvae_lstm_last.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_features": num_features,
        "num_classes": num_classes,
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
        "embedding_dim": embedding_dim,
        "max_length": max_length,
    }, last_model_path)

    scaler_path = os.path.join(save_dir, "scaler.pkl")
    encoder_path = os.path.join(save_dir, "label_encoder.pkl")
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(label_encoder, f)

    print(f"\nEntrenamiento finalizado.")
    print(f"Modelo (último): {last_model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Label Encoder: {encoder_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar CVAE-LSTM")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1.0)
    return parser.parse_args()

    
if __name__ == "__main__":
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    train(
        project_root=project_root,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        hidden_size=args.hidden_size,
        latent_dim=args.latent_dim,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        beta=args.beta,
    )