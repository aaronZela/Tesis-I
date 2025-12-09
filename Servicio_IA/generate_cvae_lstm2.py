#FINAL CVAELSTM ULTIMO c
import os
import torch
import pickle
import numpy as np
import pandas as pd
from torch import nn
from typing import Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Clase CVAELSTM del entrenamiento (copiada de CVAE-LSTM2.py)
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

    def generate(
        self, 
        z: torch.Tensor, 
        paso_labels: torch.Tensor,
        genero_labels: torch.Tensor,
        danza_labels: torch.Tensor,
        seq_len: int
    ) -> torch.Tensor:
        """Método para generar directamente desde z latente"""
        return self.decode(z, paso_labels, genero_labels, danza_labels, seq_len)

def generate_cvae_sample(
    model_path='../Servicio_IA_Entrenada/cvae_lstm_best.pt',
    seq_length=100,
    paso=1,
    genero='Hombre',
    danza='Carnaval',
    fps=30
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Cargar checkpoint del modelo
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print(f"Checkpoint cargado: {model_path}")
    
    # Instanciar modelo
    model = CVAELSTM(
        num_features=checkpoint['num_features'],
        num_pasos=checkpoint['num_pasos'],
        num_generos=checkpoint['num_generos'],
        num_danzas=checkpoint['num_danzas'],
        hidden_size=checkpoint['hidden_size'],
        latent_dim=checkpoint['latent_dim'],
        embedding_dim=checkpoint['embedding_dim'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Cargar preprocesadores desde el checkpoint
    scaler = checkpoint['scaler']
    paso_encoder = checkpoint['paso_encoder']
    genero_encoder = checkpoint['genero_encoder']
    danza_encoder = checkpoint['danza_encoder']
    
    print(f"Encoders cargados:")
    print(f"  Pasos: {paso_encoder.classes_}")
    print(f"  Géneros: {genero_encoder.classes_}")
    print(f"  Danzas: {danza_encoder.classes_}")
    
    # Codificar etiquetas
    paso_code = torch.tensor([paso_encoder.transform([str(paso)])[0]], dtype=torch.long).to(device)
    genero_code = torch.tensor([genero_encoder.transform([genero])[0]], dtype=torch.long).to(device)
    danza_code = torch.tensor([danza_encoder.transform([danza])[0]], dtype=torch.long).to(device)
    
    print(f"Generando: Paso {paso} - {genero} - {danza}")
    
    # Generar z latente (ruido)
    z = torch.randn(1, checkpoint['latent_dim']).to(device)
    
    # Generar secuencia
    with torch.no_grad():
        generated = model.generate(z, paso_code, genero_code, danza_code, seq_length)
    
    # Desnormalizar
    generated_np = generated.cpu().numpy().reshape(-1, checkpoint['num_features'])
    generated_original = scaler.inverse_transform(generated_np)
    
    # Nombres de keypoints de MediaPipe Pose (33 puntos)
    keypoint_names = [
        'NOSE', 'LEFT_EYE_INNER', 'LEFT_EYE', 'LEFT_EYE_OUTER',
        'RIGHT_EYE_INNER', 'RIGHT_EYE', 'RIGHT_EYE_OUTER',
        'LEFT_EAR', 'RIGHT_EAR', 'MOUTH_LEFT', 'MOUTH_RIGHT',
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_PINKY', 'RIGHT_PINKY',
        'LEFT_INDEX', 'RIGHT_INDEX', 'LEFT_THUMB', 'RIGHT_THUMB',
        'LEFT_HIP', 'RIGHT_HIP', 'LEFT_KNEE', 'RIGHT_KNEE',
        'LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_HEEL', 'RIGHT_HEEL',
        'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX'
    ]
    
    # Crear DataFrame con estructura MediaPipe
    df = pd.DataFrame()
    
    # Columnas iniciales
    df['frame'] = range(seq_length)
    df['timestamp'] = df['frame'] / fps
    
    # Rellenar coordenadas x, y, z y visibility para cada keypoint
    for i, kp_name in enumerate(keypoint_names):
        df[f'{kp_name}_x'] = generated_original[:, i*3]
        df[f'{kp_name}_y'] = generated_original[:, i*3 + 1]
        df[f'{kp_name}_z'] = generated_original[:, i*3 + 2]
        # Visibility alto para datos sintéticos (indica confianza)
        df[f'{kp_name}_visibility'] = 0.95
    
    # Agregar columnas quality al final
    for kp_name in keypoint_names:
        # Quality alto para datos sintéticos
        df[f'{kp_name}_quality'] = 0.90
    
    # Guardar CSV con estructura MediaPipe
    output_path = f'../Danza_Nueva/generated_cvae_paso{paso}_{genero}_{danza}.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ Secuencia CSV guardada en: {output_path}")

    # Guardar metadata separada
    import json
    metadata = {
        'paso': paso,
        'genero': genero,
        'danza': danza,
        'seq_length': seq_length,
        'fps': fps,
        'model_path': model_path
    }
    metadata_path = f'../Danza_Nueva/generated_cvae_paso{paso}_{genero}_{danza}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata guardada en: {metadata_path}")

    # Generar JSON para visualización 3D
    json_data = {
        "metadata": metadata,
        "frames": []
    }

    for idx, row in df.iterrows():
        frame_data = {
            "frame": int(row['frame']),
            "timestamp": float(row['timestamp']),
            "keypoints": []
        }
        for kp_name in keypoint_names:
            frame_data["keypoints"].append({
                "name": kp_name,
                "x": float(row[f'{kp_name}_x']),
                "y": float(row[f'{kp_name}_y']),
                "z": float(row[f'{kp_name}_z']),
                "visibility": float(row[f'{kp_name}_visibility'])
            })
        json_data["frames"].append(frame_data)

    json_path = f'../Danza_Nueva/generated_cvae_paso{paso}_{genero}_{danza}.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ JSON 3D guardado en: {json_path}")
    
    return df, json_data


def classify_labels_from_keypoints(keypoints_df, model, scaler, paso_encoder, genero_encoder, danza_encoder, device):
    """
    Clasifica las etiquetas (paso, genero, danza) a partir de keypoints extraídos de un video.
    Usa el modelo CVAE para encontrar las etiquetas que minimizan el error de reconstrucción.
    """
    # Preparar los datos de keypoints (asumiendo que keypoints_df tiene las columnas x,y,z por keypoint)
    # Extraer solo las coordenadas (ignorar frame, timestamp, visibility, quality)
    coord_cols = [col for col in keypoints_df.columns if col.endswith('_x') or col.endswith('_y') or col.endswith('_z')]
    keypoints_data = keypoints_df[coord_cols].values

    # Normalizar
    keypoints_normalized = scaler.transform(keypoints_data)

    # Convertir a tensor
    x = torch.tensor(keypoints_normalized, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, num_features)

    seq_len = x.size(1)

    # Probar todas las combinaciones de etiquetas
    best_loss = float('inf')
    best_paso = None
    best_genero = None
    best_danza = None

    for paso_str in paso_encoder.classes_:
        paso_code = torch.tensor([paso_encoder.transform([paso_str])[0]], dtype=torch.long).to(device)
        for genero in genero_encoder.classes_:
            genero_code = torch.tensor([genero_encoder.transform([genero])[0]], dtype=torch.long).to(device)
            for danza in danza_encoder.classes_:
                danza_code = torch.tensor([danza_encoder.transform([danza])[0]], dtype=torch.long).to(device)

                # Codificar
                mu, logvar = model.encode(x, paso_code, genero_code, danza_code)
                z = model.reparameterize(mu, logvar)

                # Reconstruir
                recon = model.decode(z, paso_code, genero_code, danza_code, seq_len)

                # Calcular pérdida (MSE)
                loss = nn.functional.mse_loss(recon, x)

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_paso = int(paso_str)
                    best_genero = genero
                    best_danza = danza

    return best_paso, best_genero, best_danza


def generate_from_video(keypoints_df, model_path='../Servicio_IA_Entrenada/cvae_lstm_best.pt', fps=30):
    """
    Genera una nueva secuencia de danza a partir de keypoints de un video subido.
    Clasifica las etiquetas, incrementa el paso, y genera una nueva secuencia.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar modelo y preprocesadores (igual que en generate_cvae_sample)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = CVAELSTM(
        num_features=checkpoint['num_features'],
        num_pasos=checkpoint['num_pasos'],
        num_generos=checkpoint['num_generos'],
        num_danzas=checkpoint['num_danzas'],
        hidden_size=checkpoint['hidden_size'],
        latent_dim=checkpoint['latent_dim'],
        embedding_dim=checkpoint['embedding_dim'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    scaler = checkpoint['scaler']
    paso_encoder = checkpoint['paso_encoder']
    genero_encoder = checkpoint['genero_encoder']
    danza_encoder = checkpoint['danza_encoder']

    # Clasificar etiquetas
    paso, genero, danza = classify_labels_from_keypoints(keypoints_df, model, scaler, paso_encoder, genero_encoder, danza_encoder, device)

    print(f"Etiquetas clasificadas: Paso {paso} - {genero} - {danza}")

    # Incrementar el paso para generar una nueva variación
    new_paso = paso + 1
    # Si excede el máximo, wrap around o mantener
    if str(new_paso) not in paso_encoder.classes_:
        new_paso = paso  # Mantener el mismo si no hay siguiente

    print(f"Generando nueva secuencia: Paso {new_paso} - {genero} - {danza}")

    # Generar con las nuevas etiquetas
    return generate_cvae_sample(
        model_path=model_path,
        seq_length=keypoints_df.shape[0],  # Usar la misma longitud que el video
        paso=new_paso,
        genero=genero,
        danza=danza,
        fps=fps
    )


if __name__ == "__main__":
    # Configuración general del video
    DURACION_SEGUNDOS = 45  # Puedes cambiarlo según lo necesites
    FPS = 30
    SEQ_LENGTH = DURACION_SEGUNDOS * FPS

    print("=" * 60)
    print(f"GENERADOR DE DANZAS CVAE-LSTM")
    print("=" * 60)

    # El usuario elige lo que quiere generar
    paso = int(input("👉 Ingresa el número de paso: "))
    genero = input("👉 Ingresa el género (Hombre/Mujer): ").strip().capitalize()
    danza = input("👉 Ingresa el tipo de danza (Carnaval, Morenada, Tinku, etc.): ").strip().capitalize()

    print(f"\nGenerando nueva secuencia para:")
    print(f"  • Paso: {paso}")
    print(f"  • Género: {genero}")
    print(f"  • Danza: {danza}")
    print("=" * 60)

    # Llamar al generador
    df_generated, json_data = generate_cvae_sample(
        model_path='../Servicio_IA_Entrenada/cvae_lstm_best.pt',
        seq_length=SEQ_LENGTH,
        paso=paso,
        genero=genero,
        danza=danza,
        fps=FPS
    )

    print("\n" + "-" * 60)
    print("RESUMEN DE GENERACIÓN")
    print("-" * 60)
    print(f"✓ Frames generados: {len(df_generated)}")
    print(f"✓ Duración: {df_generated['timestamp'].max():.2f} segundos")
    print(f"✓ Columnas totales: {len(df_generated.columns)}")
    print(f"✓ Tamaño aproximado del archivo: ~{len(df_generated) * len(df_generated.columns) * 8 / 1024:.2f} KB")
    print(f"\nPrimeras 3 filas (primeras 8 columnas):")
    print(df_generated.iloc[:3, :8])

    print("\n" + "=" * 60)
    print("✓ GENERACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 60)