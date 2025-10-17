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
        dropout: float = 0.5,
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
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Dropout adicional para la salida del LSTM
        self.dropout = nn.Dropout(dropout)
        
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size, num_features),
            nn.Tanh()  # Normalizar salida entre [-1, 1]
        )

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
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
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
        dropout: float = 0.5,
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
            dropout=dropout if num_layers > 1 else 0,
        )
        
        # Dropout adicional para la salida del LSTM
        self.dropout = nn.Dropout(dropout)
        
        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
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
        
        # Apply dropout
        last_hidden = self.dropout(last_hidden)
        
        # Output layer
        output = self.output_fc(last_hidden)
        
        return output


def generate_samples(
    generator: nn.Module,
    device: torch.device,
    num_samples: int,
    seq_length: int,
    noise_dim: int,
    label_encoder: LabelEncoder,
    dance_class: str = None,
) -> np.ndarray:
    """
    Genera muestras sintéticas usando el generador entrenado.
    
    Args:
        generator: Modelo generador
        device: Dispositivo (CPU/GPU)
        num_samples: Número de muestras a generar
        seq_length: Longitud de las secuencias
        noise_dim: Dimensión del ruido
        label_encoder: Encoder de etiquetas
        dance_class: Clase específica de danza (opcional)
    
    Returns:
        Array numpy con las muestras generadas
    """
    generator.eval()
    
    with torch.no_grad():
        # Generar ruido
        noise = torch.randn(num_samples, noise_dim).to(device)
        
        # Generar etiquetas
        if dance_class is not None:
            label_idx = label_encoder.transform([dance_class])[0]
            labels = torch.full((num_samples,), label_idx, dtype=torch.long).to(device)
        else:
            # Generar etiquetas aleatorias
            labels = torch.randint(0, len(label_encoder.classes_), (num_samples,)).to(device)
        
        # Generar muestras
        generated = generator(noise, labels, seq_length)
        
        return generated.cpu().numpy()


def train_gan(
    project_root: str,
    max_length: int = 150,
    batch_size: int = 2,
    epochs: int = 50,
    noise_dim: int = 64,
    hidden_size: int = 256,
    num_layers: int = 1,
    dropout: float = 0.5,
    lr_g: float = 2e-4,
    lr_d: float = 2e-4,
    beta1: float = 0.5,
    d_rounds: int = 1,
    g_rounds: int = 1,
    label_smoothing: float = 0.1,
    gradient_penalty_weight: float = 0.0,
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
    print(f"Número de características: {num_features}")

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
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    discriminator = Discriminator(
        num_features=num_features,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
    ).to(device)

    print(f"\nGenerator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

    # Initialize optimizers con learning rates separados
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_g, betas=(beta1, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, 0.999))

    # Loss function
    criterion = nn.BCELoss()

    # Training history
    history = {
        'g_loss': [],
        'd_loss': [],
        'val_g_loss': [],
        'val_d_loss': [],
    }

    # Training loop
    best_val_g_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        generator.train()
        discriminator.train()
        
        g_losses = []
        d_losses = []
        
        for batch_idx, (real_data, labels, lengths) in enumerate(train_loader):
            batch_size_actual = real_data.size(0)
            real_data = real_data.to(device)
            labels = labels.to(device)
            
            # Label smoothing para mejorar estabilidad
            real_labels = torch.ones(batch_size_actual, 1).to(device) * (1.0 - label_smoothing)
            fake_labels = torch.zeros(batch_size_actual, 1).to(device) + label_smoothing
            
            # Train Discriminator (d_rounds veces)
            for _ in range(d_rounds):
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
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
                
                d_optimizer.step()
            
            # Train Generator (g_rounds veces)
            for _ in range(g_rounds):
                g_optimizer.zero_grad()
                
                noise = torch.randn(batch_size_actual, noise_dim).to(device)
                fake_data = generator(noise, labels, real_data.size(1))
                fake_output = discriminator(fake_data, labels)
                
                # Generator quiere que el discriminador clasifique como real (1)
                g_loss = criterion(fake_output, torch.ones(batch_size_actual, 1).to(device))
                g_loss.backward()
                
                # Gradient clipping para estabilidad
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                
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
        
        # Guardar historial
        history['g_loss'].append(avg_g_loss)
        history['d_loss'].append(avg_d_loss)
        history['val_g_loss'].append(avg_val_g_loss)
        history['val_d_loss'].append(avg_val_d_loss)
        
        print(f"Epoch {epoch:03d}/{epochs} | "
              f"G: {avg_g_loss:.6f} | D: {avg_d_loss:.6f} | "
              f"Val G: {avg_val_g_loss:.6f} | Val D: {avg_val_d_loss:.6f}")
        
        # Guardar mejor modelo
        if avg_val_g_loss < best_val_g_loss:
            best_val_g_loss = avg_val_g_loss
            save_dir = os.path.join(project_root, "Servicio_IA_Entrenada")
            os.makedirs(save_dir, exist_ok=True)
            
            torch.save({
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "noise_dim": noise_dim,
                "num_features": num_features,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "num_classes": num_classes,
                "max_length": max_length,
                "dropout": dropout,
                "scaler": scaler,
                "label_encoder": label_encoder,
                "epoch": epoch,
                "best_val_g_loss": best_val_g_loss,
            }, os.path.join(save_dir, "gan_lstm_best_model.pt"))
            
            print(f"  -> Mejor modelo guardado (Val G Loss: {best_val_g_loss:.6f})")

    # Save final model
    save_dir = os.path.join(project_root, "Servicio_IA_Entrenada")
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save({
        "generator_state_dict": generator.state_dict(),
        "discriminator_state_dict": discriminator.state_dict(),
        "noise_dim": noise_dim,
        "num_features": num_features,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "max_length": max_length,
        "dropout": dropout,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "training_history": history,
    }, os.path.join(save_dir, "gan_lstm_final_model.pt"))
    
    # Guardar historial de entrenamiento
    import json
    with open(os.path.join(save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Entrenamiento GAN-LSTM finalizado.")
    print(f"Mejor Val G Loss: {best_val_g_loss:.6f}")
    print(f"Modelos guardados en: {save_dir}")
    print(f"  - gan_lstm_best_model.pt (mejor modelo)")
    print(f"  - gan_lstm_final_model.pt (modelo final)")
    print(f"  - training_history.json (historial)")
    print(f"{'='*60}")


def load_trained_generator(model_path: str, device: torch.device = None):
    """
    Carga un generador entrenado desde un archivo.
    
    Args:
        model_path: Ruta al archivo del modelo
        device: Dispositivo (CPU/GPU)
    
    Returns:
        Tupla (generator, metadata_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    generator = Generator(
        noise_dim=checkpoint['noise_dim'],
        num_features=checkpoint['num_features'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint.get('num_layers', 1),
        num_classes=checkpoint['num_classes'],
        dropout=checkpoint.get('dropout', 0.5),
    ).to(device)
    
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    
    metadata = {
        'scaler': checkpoint['scaler'],
        'label_encoder': checkpoint['label_encoder'],
        'max_length': checkpoint['max_length'],
        'noise_dim': checkpoint['noise_dim'],
        'num_features': checkpoint['num_features'],
    }
    
    return generator, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar GAN-LSTM mejorado")
    parser.add_argument("--max_length", type=int, default=150, help="Longitud máxima de secuencia")
    parser.add_argument("--batch_size", type=int, default=2, help="Tamaño del batch")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--noise_dim", type=int, default=64, help="Dimensión del ruido")
    parser.add_argument("--hidden_size", type=int, default=256, help="Tamaño hidden del LSTM")
    parser.add_argument("--num_layers", type=int, default=1, help="Número de capas LSTM")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--lr_g", type=float, default=2e-4, help="Learning rate del generador")
    parser.add_argument("--lr_d", type=float, default=2e-4, help="Learning rate del discriminador")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 para Adam")
    parser.add_argument("--d_rounds", type=int, default=1, help="Rondas de entrenamiento del discriminador")
    parser.add_argument("--g_rounds", type=int, default=1, help="Rondas de entrenamiento del generador")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
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
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        beta1=args.beta1,
        d_rounds=args.d_rounds,
        g_rounds=args.g_rounds,
        label_smoothing=args.label_smoothing,
    )