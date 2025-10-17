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
        scaler: StandardScaler = None,
        label_encoder: LabelEncoder = None,
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


class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        num_features: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.class_embedding = nn.Embedding(num_classes, 32)
        
        self.lstm = nn.LSTM(
            input_size=noise_dim + 32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.output_fc = nn.Linear(hidden_size, num_features)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor, seq_len: int) -> torch.Tensor:
        batch_size = noise.size(0)
        
        # Expand noise to sequence length
        noise_seq = noise.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Get class embeddings
        class_emb = self.class_embedding(labels).unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate noise and class embeddings
        lstm_input = torch.cat([noise_seq, class_emb], dim=-1)
        
        # Generate sequence
        lstm_out, _ = self.lstm(lstm_input)
        
        # Output layer
        generated = self.output_fc(lstm_out)
        
        return generated


class Discriminator(nn.Module):
    def __init__(
        self,
        num_features: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.class_embedding = nn.Embedding(num_classes, 32)
        
        self.lstm = nn.LSTM(
            input_size=num_features + 32,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Get class embeddings
        class_emb = self.class_embedding(labels).unsqueeze(1).repeat(1, x.size(1), 1)
        
        # Concatenate input and class embeddings
        lstm_input = torch.cat([x, class_emb], dim=-1)
        
        # Process sequence
        lstm_out, (h_n, _) = self.lstm(lstm_input)
        
        # Use last hidden state for classification
        last_hidden = h_n[-1]
        
        # Output layer
        output = self.output_fc(last_hidden)
        
        return output


def train_gan(
    project_root: str,
    max_length: int = 150,
    batch_size: int = 2,
    epochs: int = 50,
    noise_dim: int = 64,
    hidden_size: int = 256,
    lr: float = 2e-4,
    beta1: float = 0.5,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Discover CSV files
    coord_dir = os.path.join(project_root, "Coordenadas_csv")
    csv_paths = sorted(glob.glob(os.path.join(coord_dir, "*_processed.csv")))
    
    if len(csv_paths) == 0:
        raise FileNotFoundError(f"No se encontraron CSV procesados en: {coord_dir}")
    
    print(f"Encontrados {len(csv_paths)} archivos CSV")

    # Create dataset
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
    num_features = full_dataset_for_scaler.num_features
    
    print(f"Danzas encontradas: {label_encoder.classes_}")

    dataset = StepDataset(
        csv_paths=csv_paths,
        max_length=max_length,
        scaler=scaler,
        label_encoder=label_encoder,
        fit_scaler=False,
        fit_encoder=False,
    )

    # Split dataset
    val_split = 0.2
    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    print(f"Dataset total: {len(dataset)}, Train: {train_len}, Val: {val_len}")

    # Create data loaders
    effective_batch_size = min(batch_size, train_len)
    train_loader = DataLoader(train_ds, batch_size=effective_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, val_len), shuffle=False, drop_last=True)

    # Initialize models
    generator = Generator(
        noise_dim=noise_dim,
        num_features=num_features,
        hidden_size=hidden_size,
        num_classes=num_classes,
    ).to(device)

    discriminator = Discriminator(
        num_features=num_features,
        hidden_size=hidden_size,
        num_classes=num_classes,
    ).to(device)

    # Initialize optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        g_losses = []
        d_losses = []
        
        for batch_idx, (real_data, labels, lengths) in enumerate(train_loader):
            batch_size_actual = real_data.size(0)
            real_data = real_data.to(device)
            labels = labels.to(device)
            
            # Create labels for loss calculation
            real_labels = torch.ones(batch_size_actual, 1).to(device)
            fake_labels = torch.zeros(batch_size_actual, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            # Real data
            real_output = discriminator(real_data, labels)
            d_loss_real = criterion(real_output, real_labels)
            
            # Fake data
            noise = torch.randn(batch_size_actual, noise_dim).to(device)
            fake_data = generator(noise, labels, real_data.size(1))
            fake_output = discriminator(fake_data.detach(), labels)
            d_loss_fake = criterion(fake_output, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            
            fake_output = discriminator(fake_data, labels)
            g_loss = criterion(fake_output, real_labels)
            g_loss.backward()
            g_optimizer.step()
            
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

        # Validation
        generator.eval()
        discriminator.eval()
        
        val_g_losses = []
        val_d_losses = []
        
        with torch.no_grad():
            for real_data, labels, lengths in val_loader:
                batch_size_actual = real_data.size(0)
                real_data = real_data.to(device)
                labels = labels.to(device)
                
                real_labels = torch.ones(batch_size_actual, 1).to(device)
                fake_labels = torch.zeros(batch_size_actual, 1).to(device)
                
                # Discriminator loss
                real_output = discriminator(real_data, labels)
                d_loss_real = criterion(real_output, real_labels)
                
                noise = torch.randn(batch_size_actual, noise_dim).to(device)
                fake_data = generator(noise, labels, real_data.size(1))
                fake_output = discriminator(fake_data, labels)
                d_loss_fake = criterion(fake_output, fake_labels)
                
                d_loss = d_loss_real + d_loss_fake
                
                # Generator loss
                fake_output = discriminator(fake_data, labels)
                g_loss = criterion(fake_output, real_labels)
                
                val_g_losses.append(g_loss.item())
                val_d_losses.append(d_loss.item())

        avg_g_loss = np.mean(g_losses)
        avg_d_loss = np.mean(d_losses)
        avg_val_g_loss = np.mean(val_g_losses)
        avg_val_d_loss = np.mean(val_d_losses)
        
        print(f"Epoch {epoch:03d} | G: {avg_g_loss:.6f} | D: {avg_d_loss:.6f} | "
              f"Val G: {avg_val_g_loss:.6f} | Val D: {avg_val_d_loss:.6f}")

    # Save models
    save_dir = os.path.join(project_root, "Servicio_IA_Entrenada")
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "noise_dim": noise_dim,
        "num_features": num_features,
        "hidden_size": hidden_size,
        "num_classes": num_classes,
        "max_length": max_length,
        "scaler": scaler,
        "label_encoder": label_encoder,
    }, os.path.join(save_dir, "gan_lstm_model.pt"))
    
    print(f"\nEntrenamiento GAN finalizado.")
    print(f"Modelo guardado en: {os.path.join(save_dir, 'gan_lstm_model.pt')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar GAN-LSTM")
    parser.add_argument("--max_length", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--noise_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    train_gan(
        project_root=project_root,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        noise_dim=args.noise_dim,
        hidden_size=args.hidden_size,
        lr=args.lr,
        beta1=args.beta1,
    )