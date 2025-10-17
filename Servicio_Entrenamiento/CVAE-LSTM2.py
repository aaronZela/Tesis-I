import os
import glob
import argparse
import pickle
import re
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
        scaler: StandardScaler = None,
        paso_encoder: LabelEncoder = None,
        genero_encoder: LabelEncoder = None,
        danza_encoder: LabelEncoder = None,
        fit_scaler: bool = False,
        fit_encoders: bool = False,
    ) -> None:
        self.max_length = max_length
        self.scaler = scaler or StandardScaler()
        self.paso_encoder = paso_encoder or LabelEncoder()
        self.genero_encoder = genero_encoder or LabelEncoder()
        self.danza_encoder = danza_encoder or LabelEncoder()

        self.sequences: List[np.ndarray] = []
        self.paso_labels: List[int] = []
        self.genero_labels: List[str] = []
        self.danza_labels: List[str] = []
        self.lengths: List[int] = []

        all_features = []
        pasos = []
        generos = []
        danzas = []

        for path in csv_paths:
            df = pd.read_csv(path)
            feature_cols = [c for c in df.columns if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")]
            if len(feature_cols) == 0:
                continue

            features = df[feature_cols].values.astype(np.float32)
            
            # Parsear nombre del archivo: "Paso 1 - Hombre - Carnaval_processed.csv"
            basename = os.path.basename(path).replace("_processed.csv", "")
            parsed = self._parse_filename(basename)
            
            if parsed is None:
                print(f"⚠️ No se pudo parsear: {basename}, omitiendo...")
                continue
            
            paso, genero, danza = parsed
            
            all_features.append(features)
            pasos.append(paso)
            generos.append(genero)
            danzas.append(danza)

        if len(all_features) == 0:
            raise ValueError("No se encontraron características en los CSV.")

        if fit_scaler:
            concat_features = np.vstack(all_features)
            self.scaler.fit(concat_features)

        if fit_encoders:
            self.paso_encoder.fit([str(p) for p in pasos])
            self.genero_encoder.fit(generos)
            self.danza_encoder.fit(danzas)

        for features, paso, genero, danza in zip(all_features, pasos, generos, danzas):
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
            self.paso_labels.append(paso)
            self.genero_labels.append(genero)
            self.danza_labels.append(danza)
            self.lengths.append(actual_len)

        self.num_features = self.sequences[0].shape[1]

    def _parse_filename(self, basename: str) -> Tuple[int, str, str] | None:
        """
        Parsea nombres como:
        - "Paso 1 - Hombre - Carnaval"
        - "Paso 2 - Mujer - Turcos"
        Retorna: (numero_paso, genero, danza)
        """
        # Patrón: Paso X - Género - Danza
        pattern = r"Paso\s+(\d+)\s*-\s*(\w+)\s*-\s*(.+)"
        match = re.match(pattern, basename, re.IGNORECASE)
        
        if match:
            paso_num = int(match.group(1))
            genero = match.group(2).strip().capitalize()
            danza = match.group(3).strip().capitalize()
            return paso_num, genero, danza
        
        return None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq = torch.from_numpy(self.sequences[idx])
        
        paso_label = torch.tensor(
            self.paso_encoder.transform([str(self.paso_labels[idx])])[0], 
            dtype=torch.long
        )
        genero_label = torch.tensor(
            self.genero_encoder.transform([self.genero_labels[idx]])[0], 
            dtype=torch.long
        )
        danza_label = torch.tensor(
            self.danza_encoder.transform([self.danza_labels[idx]])[0], 
            dtype=torch.long
        )
        length = torch.tensor(self.lengths[idx], dtype=torch.long)
        
        return seq, paso_label, genero_label, danza_label, length


class CVAELSTM(nn.Module):
    def __init__(
        self,
        num_features: int,
        num_pasos: int,
        num_generos: int,
        num_danzas: int,
        hidden_size: int = 256,
        latent_dim: int = 64,
        num_layers: int = 1,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_pasos = num_pasos
        self.num_generos = num_generos
        self.num_danzas = num_danzas
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # Embeddings separados para cada categoría
        self.paso_embedding = nn.Embedding(num_pasos, embedding_dim)
        self.genero_embedding = nn.Embedding(num_generos, embedding_dim)
        self.danza_embedding = nn.Embedding(num_danzas, embedding_dim)
        
        # Total embedding dimension
        total_emb_dim = 3 * embedding_dim
        
        # Encoder
        self.encoder_lstm = nn.LSTM(
            input_size=num_features + total_emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_logvar = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim + total_emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_fc = nn.Linear(hidden_size, num_features)

    def _get_combined_embedding(
        self, 
        paso_labels: torch.Tensor, 
        genero_labels: torch.Tensor, 
        danza_labels: torch.Tensor
    ) -> torch.Tensor:
        """Combina los 3 embeddings en uno solo"""
        paso_emb = self.paso_embedding(paso_labels)
        genero_emb = self.genero_embedding(genero_labels)
        danza_emb = self.danza_embedding(danza_labels)
        return torch.cat([paso_emb, genero_emb, danza_emb], dim=-1)

    def encode(
        self, 
        x: torch.Tensor, 
        paso_labels: torch.Tensor,
        genero_labels: torch.Tensor,
        danza_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined_emb = self._get_combined_embedding(paso_labels, genero_labels, danza_labels)
        combined_emb = combined_emb.unsqueeze(1).repeat(1, x.size(1), 1)
        x_cond = torch.cat([x, combined_emb], dim=-1)
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

    def decode(
        self, 
        z: torch.Tensor, 
        paso_labels: torch.Tensor,
        genero_labels: torch.Tensor,
        danza_labels: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        combined_emb = self._get_combined_embedding(paso_labels, genero_labels, danza_labels)
        combined_emb = combined_emb.unsqueeze(1).repeat(1, seq_len, 1)
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)
        z_cond = torch.cat([z_seq, combined_emb], dim=-1)
        out, _ = self.decoder_lstm(z_cond)
        recon = self.output_fc(out)
        return recon

    def forward(
        self, 
        x: torch.Tensor, 
        paso_labels: torch.Tensor,
        genero_labels: torch.Tensor,
        danza_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, paso_labels, genero_labels, danza_labels)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, paso_labels, genero_labels, danza_labels, x.size(1))
        return recon, mu, logvar


def loss_function(
    recon_x: torch.Tensor, 
    x: torch.Tensor, 
    mu: torch.Tensor, 
    logvar: torch.Tensor, 
    lengths: torch.Tensor, 
    beta: float = 1.0
) -> torch.Tensor:
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
    embedding_dim: int = 16,
    lr: float = 1e-3,
    beta: float = 1.0,
    val_split: float = 0.2,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Usando dispositivo: {device}")

    csv_paths = discover_processed_csvs(project_root)
    print(f"📁 Encontrados {len(csv_paths)} archivos CSV")

    # Crear dataset inicial para ajustar encoders y scaler
    full_dataset_for_scaler = StepDataset(
        csv_paths=csv_paths,
        max_length=max_length,
        scaler=None,
        paso_encoder=None,
        genero_encoder=None,
        danza_encoder=None,
        fit_scaler=True,
        fit_encoders=True,
    )

    scaler = full_dataset_for_scaler.scaler
    paso_encoder = full_dataset_for_scaler.paso_encoder
    genero_encoder = full_dataset_for_scaler.genero_encoder
    danza_encoder = full_dataset_for_scaler.danza_encoder
    
    num_pasos = len(paso_encoder.classes_)
    num_generos = len(genero_encoder.classes_)
    num_danzas = len(danza_encoder.classes_)
    
    print(f"\n📊 Categorías detectadas:")
    print(f"   Pasos: {list(paso_encoder.classes_)}")
    print(f"   Géneros: {list(genero_encoder.classes_)}")
    print(f"   Danzas: {list(danza_encoder.classes_)}")

    # Crear dataset final con encoders ya ajustados
    dataset = StepDataset(
        csv_paths=csv_paths,
        max_length=max_length,
        scaler=scaler,
        paso_encoder=paso_encoder,
        genero_encoder=genero_encoder,
        danza_encoder=danza_encoder,
        fit_scaler=False,
        fit_encoders=False,
    )

    num_features = dataset.num_features

    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    print(f"\n📦 Dataset total: {len(dataset)}, Train: {train_len}, Val: {val_len}")

    effective_batch_size = min(batch_size, train_len)
    print(f"🔢 Batch size ajustado: {effective_batch_size}")

    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, val_len), shuffle=False, drop_last=False)

    model = CVAELSTM(
        num_features=num_features,
        num_pasos=num_pasos,
        num_generos=num_generos,
        num_danzas=num_danzas,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
        num_layers=1,
        embedding_dim=embedding_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    save_dir = os.path.join(project_root, "Servicio_IA_Entrenada")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n🚀 Iniciando entrenamiento...\n")

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        
        for seq, paso_labels, genero_labels, danza_labels, lengths in train_loader:
            seq = seq.to(device)
            paso_labels = paso_labels.to(device)
            genero_labels = genero_labels.to(device)
            danza_labels = danza_labels.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            recon, mu, logvar = model(seq, paso_labels, genero_labels, danza_labels)
            loss = loss_function(recon, seq, mu, logvar, lengths, beta=beta)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for seq, paso_labels, genero_labels, danza_labels, lengths in val_loader:
                seq = seq.to(device)
                paso_labels = paso_labels.to(device)
                genero_labels = genero_labels.to(device)
                danza_labels = danza_labels.to(device)
                lengths = lengths.to(device)
                
                recon, mu, logvar = model(seq, paso_labels, genero_labels, danza_labels)
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
                "num_pasos": num_pasos,
                "num_generos": num_generos,
                "num_danzas": num_danzas,
                "hidden_size": hidden_size,
                "latent_dim": latent_dim,
                "embedding_dim": embedding_dim,
                "max_length": max_length,
                "scaler": scaler,
                "paso_encoder": paso_encoder,
                "genero_encoder": genero_encoder,
                "danza_encoder": danza_encoder,
            }, model_path)
            print(f"💾 Guardado mejor modelo (val_loss={val_loss:.6f})")

    # Guardar modelo final
    last_model_path = os.path.join(save_dir, "cvae_lstm_last.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_features": num_features,
        "num_pasos": num_pasos,
        "num_generos": num_generos,
        "num_danzas": num_danzas,
        "hidden_size": hidden_size,
        "latent_dim": latent_dim,
        "embedding_dim": embedding_dim,
        "max_length": max_length,
    }, last_model_path)

    # Guardar encoders y scaler
    scaler_path = os.path.join(save_dir, "scaler.pkl")
    paso_encoder_path = os.path.join(save_dir, "paso_encoder.pkl")
    genero_encoder_path = os.path.join(save_dir, "genero_encoder.pkl")
    danza_encoder_path = os.path.join(save_dir, "danza_encoder.pkl")
    
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(paso_encoder_path, "wb") as f:
        pickle.dump(paso_encoder, f)
    with open(genero_encoder_path, "wb") as f:
        pickle.dump(genero_encoder, f)
    with open(danza_encoder_path, "wb") as f:
        pickle.dump(danza_encoder, f)

    print(f"\n✅ Entrenamiento finalizado.")
    print(f"📁 Modelo (último): {last_model_path}")
    print(f"📁 Modelo (mejor): {os.path.join(save_dir, 'cvae_lstm_best.pt')}")
    print(f"📁 Encoders guardados en: {save_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar CVAE-LSTM Multi-Label")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=16)
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