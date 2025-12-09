#FINAL GANLSTM ULTIMO
import os
import torch
import pickle
import numpy as np
import pandas as pd
import json
from torch import nn
from sklearn.preprocessing import StandardScaler

# Clases Generator y Discriminator del entrenamiento (copiadas de GAN-LSTM2.py)
class Generator(nn.Module):
    def __init__(
        self,
        noise_dim: int,
        num_features: int,
        hidden_size: int = 256,
        num_layers: int = 1,
        num_pasos: int = 1,
        num_generos: int = 1,
        num_danzas: int = 1,
        embedding_dim: int = 16,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.paso_embedding = nn.Embedding(num_pasos, embedding_dim)
        self.genero_embedding = nn.Embedding(num_generos, embedding_dim)
        self.danza_embedding = nn.Embedding(num_danzas, embedding_dim)
        total_emb_dim = 3 * embedding_dim

        self.lstm = nn.LSTM(
            input_size=noise_dim + total_emb_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.output_fc = nn.Sequential(
            nn.Linear(hidden_size, num_features),
            nn.Tanh()
        )

    def forward(
        self,
        noise: torch.Tensor,
        paso_labels: torch.Tensor,
        genero_labels: torch.Tensor,
        danza_labels: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        noise_seq = noise.unsqueeze(1).repeat(1, seq_len, 1)
        paso_emb = self.paso_embedding(paso_labels).unsqueeze(1).repeat(1, seq_len, 1)
        genero_emb = self.genero_embedding(genero_labels).unsqueeze(1).repeat(1, seq_len, 1)
        danza_emb = self.danza_embedding(danza_labels).unsqueeze(1).repeat(1, seq_len, 1)
        class_emb = torch.cat([paso_emb, genero_emb, danza_emb], dim=-1)

        lstm_input = torch.cat([noise_seq, class_emb], dim=-1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = self.dropout(lstm_out)
        generated = self.output_fc(lstm_out)
        return generated


def generate_gan_sample(
    model_path='gan_lstm_best_model.pt',
    seq_length=100,
    paso=1,
    genero='Hombre',
    danza='Carnaval',
    fps=30
):
    """
    Genera una secuencia de danza usando GAN-LSTM con formato MediaPipe Pose.
    Estructura de columnas idéntica al CVAE-LSTM.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Cargar checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    print(f"Checkpoint cargado: {model_path}")

    generator = Generator(
        noise_dim=checkpoint['noise_dim'],
        num_features=checkpoint['num_features'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint.get('num_layers', 1),
        num_pasos=checkpoint['num_pasos'],
        num_generos=checkpoint['num_generos'],
        num_danzas=checkpoint['num_danzas'],
        embedding_dim=checkpoint.get('embedding_dim', 16),
        dropout=checkpoint.get('dropout', 0.5),
    ).to(device)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()

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

    # Generar ruido
    noise = torch.randn(1, checkpoint['noise_dim']).to(device)

    # Generar secuencia
    with torch.no_grad():
        generated = generator(noise, paso_code, genero_code, danza_code, seq_length)

    generated_np = generated.cpu().numpy().reshape(-1, checkpoint['num_features'])
    generated_original = scaler.inverse_transform(generated_np)

    # Keypoints de MediaPipe (33)
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

    # Crear DataFrame con misma estructura que el CVAE-LSTM
    df = pd.DataFrame()
    df['frame'] = range(seq_length)
    df['timestamp'] = df['frame'] / fps

    # Añadir coordenadas y visibility
    for i, kp_name in enumerate(keypoint_names):
        df[f'{kp_name}_x'] = generated_original[:, i * 3]
        df[f'{kp_name}_y'] = generated_original[:, i * 3 + 1]
        df[f'{kp_name}_z'] = generated_original[:, i * 3 + 2]
        df[f'{kp_name}_visibility'] = 0.95  # fijo para datos sintéticos

    # Añadir quality
    for kp_name in keypoint_names:
        df[f'{kp_name}_quality'] = 0.90

    # Guardar CSV
    output_path = f'../Danza_Nueva/generated_gan_paso{paso}_{genero}_{danza}.csv'
    df.to_csv(output_path, index=False)
    print(f"✓ CSV generado en: {output_path}")

    # Guardar metadata
    metadata = {
        'modelo': 'GAN-LSTM',
        'paso': paso,
        'genero': genero,
        'danza': danza,
        'seq_length': seq_length,
        'fps': fps,
        'model_path': model_path
    }
    metadata_path = f'../Danza_Nueva/generated_gan_paso{paso}_{genero}_{danza}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata guardada en: {metadata_path}")

    # Crear JSON 3D igual al CVAE-LSTM
    json_data = {
        "metadata": metadata,
        "frames": []
    }

    for _, row in df.iterrows():
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

    json_path = f'../Danza_Nueva/generated_gan_paso{paso}_{genero}_{danza}.json'
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✓ JSON 3D guardado en: {json_path}")

    return df, json_data


def generate_gan_from_video(keypoints_df, cvae_model_path='../Servicio_IA_Entrenada/cvae_lstm_best.pt', gan_model_path='../Servicio_IA_Entrenada/gan_lstm_best_model.pt', fps=30):
    """
    Genera una nueva secuencia de danza usando GAN-LSTM a partir de keypoints de un video subido.
    Usa el CVAE para clasificar las etiquetas del video, luego incrementa el paso y genera con GAN.
    """
    from generate_cvae_lstm2 import classify_labels_from_keypoints, CVAELSTM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Cargar CVAE para clasificación
    cvae_checkpoint = torch.load(cvae_model_path, map_location=device, weights_only=False)
    cvae_model = CVAELSTM(
        num_features=cvae_checkpoint['num_features'],
        num_pasos=cvae_checkpoint['num_pasos'],
        num_generos=cvae_checkpoint['num_generos'],
        num_danzas=cvae_checkpoint['num_danzas'],
        hidden_size=cvae_checkpoint['hidden_size'],
        latent_dim=cvae_checkpoint['latent_dim'],
        embedding_dim=cvae_checkpoint['embedding_dim'],
    ).to(device)
    cvae_model.load_state_dict(cvae_checkpoint['model_state_dict'])
    cvae_model.eval()

    scaler = cvae_checkpoint['scaler']
    paso_encoder = cvae_checkpoint['paso_encoder']
    genero_encoder = cvae_checkpoint['genero_encoder']
    danza_encoder = cvae_checkpoint['danza_encoder']

    # Clasificar etiquetas usando CVAE
    paso, genero, danza = classify_labels_from_keypoints(keypoints_df, cvae_model, scaler, paso_encoder, genero_encoder, danza_encoder, device)

    print(f"Etiquetas clasificadas con CVAE: Paso {paso} - {genero} - {danza}")

    # Incrementar el paso para generar una nueva variación
    new_paso = paso + 1
    if str(new_paso) not in paso_encoder.classes_:
        new_paso = paso  # Mantener el mismo si no hay siguiente

    print(f"Generando nueva secuencia con GAN: Paso {new_paso} - {genero} - {danza}")

    # Generar con GAN usando las etiquetas clasificadas e incrementadas
    return generate_gan_sample(
        model_path=gan_model_path,
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
    print(f"GENERADOR DE DANZAS GAN-LSTM")
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
    df_generated, json_data = generate_gan_sample(
        model_path='../Servicio_IA_Entrenada/gan_lstm_best_model.pt',
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
