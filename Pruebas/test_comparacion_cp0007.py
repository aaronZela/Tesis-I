import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. CONFIGURACIÓN
# ==========================================
# Ruta relativa a la carpeta donde están tus resultados (ajusta si es necesario)
RUTA_RESULTADOS = os.path.join(os.path.dirname(__file__), '..', 'Backend', 'Danza_Nueva')

# Lista exacta de los escenarios vistos en tu imagen
ESCENARIOS = [
    'paso1_Hombre_Carnaval',
    'paso1_Hombre_Montonero',
    'paso1_Mujer_Turco',
    'paso2_Hombre_Montonero',
    'paso3_Hombre_Carnaval'
]

# ==========================================
# 2. CLASE EVALUADORA (MÉTRICAS)
# ==========================================
class MotionEvaluator:
    def __init__(self):
        # Pares de huesos para medir consistencia (MediaPipe)
        self.bones = [
            ('LEFT_SHOULDER', 'LEFT_ELBOW'),
            ('LEFT_ELBOW', 'LEFT_WRIST'),
            ('RIGHT_SHOULDER', 'RIGHT_ELBOW'),
            ('RIGHT_ELBOW', 'RIGHT_WRIST'),
            ('LEFT_HIP', 'LEFT_KNEE'),
            ('LEFT_KNEE', 'LEFT_ANKLE'),
            ('RIGHT_HIP', 'RIGHT_KNEE'),
            ('RIGHT_KNEE', 'RIGHT_ANKLE'),
        ]

    def get_coords(self, df, kp_name):
        return df[[f'{kp_name}_x', f'{kp_name}_y', f'{kp_name}_z']].values

    def calculate_bone_variance(self, df):
        """Métrica 1: Consistencia Ósea (Menor es mejor). Mide deformaciones ilegales."""
        all_variances = []
        for joint_a, joint_b in self.bones:
            try:
                pos_a = self.get_coords(df, joint_a)
                pos_b = self.get_coords(df, joint_b)
                lengths = np.linalg.norm(pos_a - pos_b, axis=1)
                variance = np.var(lengths)
                all_variances.append(variance)
            except KeyError:
                continue
        return np.mean(all_variances) * 1000 if all_variances else 0.0

    def calculate_jerk(self, df, keypoint_names):
        """Métrica 2: Suavidad/Jerk (Menor es mejor). Mide temblores."""
        total_jerk = 0
        count = 0
        for kp in keypoint_names:
            try:
                coords = self.get_coords(df, kp)
                vel = np.diff(coords, axis=0)
                acc = np.diff(vel, axis=0)
                jerk = np.diff(acc, axis=0)
                kp_jerk = np.mean(np.linalg.norm(jerk, axis=1))
                total_jerk += kp_jerk
                count += 1
            except KeyError:
                continue
        return total_jerk / count if count > 0 else 0.0

    def calculate_amplitude(self, df, keypoint_names):
        """Métrica 3: Amplitud de Movimiento (Mayor es mejor, indica dinamismo)."""
        # Calcula la desviación estándar de la posición de cada punto (cuánto se mueve del centro)
        total_std = 0
        count = 0
        for kp in keypoint_names:
            try:
                coords = self.get_coords(df, kp)
                # std dev promedio de x, y, z
                std = np.mean(np.std(coords, axis=0))
                total_std += std
                count += 1
            except KeyError:
                continue
        return total_std / count if count > 0 else 0.0

# ==========================================
# 3. UTILIDADES JSON
# ==========================================
def json_to_dataframe(json_path):
    """Convierte el JSON jerárquico de danza a un DataFrame plano."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        flat_data = []

        for fr in frames:
            row = {'frame': fr.get('frame'), 'timestamp': fr.get('timestamp')}
            for kp in fr.get('keypoints', []):
                name = kp['name']
                row[f'{name}_x'] = kp['x']
                row[f'{name}_y'] = kp['y']
                row[f'{name}_z'] = kp['z']
                # row[f'{name}_viz'] = kp['visibility'] # Opcional
            flat_data.append(row)
        
        return pd.DataFrame(flat_data)
    except Exception as e:
        print(f"⚠️ Error leyendo {os.path.basename(json_path)}: {e}")
        return None

# ==========================================
# 4. EJECUCIÓN PRINCIPAL
# ==========================================
def ejecutar_comparacion_masiva():
    print("\n" + "="*80)
    print("COMPARACIÓN MASIVA DE MODELOS (CVAE vs GAN) - FORMATO JSON")
    print("="*80)
    
    evaluator = MotionEvaluator()
    results_summary = []
    
    # Keypoints principales para evaluar
    kps_eval = [
        'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
        'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP', 
        'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE'
    ]

    for escenario in ESCENARIOS:
        print(f"\n📂 Analizando: {escenario} ...")
        
        # Construir nombres de archivo
        file_cvae = f"generated_cvae_{escenario}.json"
        file_gan = f"generated_gan_{escenario}.json"
        
        path_cvae = os.path.join(RUTA_RESULTADOS, file_cvae)
        path_gan = os.path.join(RUTA_RESULTADOS, file_gan)
        
        # Cargar datos
        df_cvae = json_to_dataframe(path_cvae)
        df_gan = json_to_dataframe(path_gan)
        
        if df_cvae is None or df_gan is None:
            print(f"   ❌ Saltando escenario por falta de archivos.")
            continue

        # Calcular Métricas CVAE
        c_bone = evaluator.calculate_bone_variance(df_cvae)
        c_jerk = evaluator.calculate_jerk(df_cvae, kps_eval)
        c_amp = evaluator.calculate_amplitude(df_cvae, kps_eval)
        
        # Calcular Métricas GAN
        g_bone = evaluator.calculate_bone_variance(df_gan)
        g_jerk = evaluator.calculate_jerk(df_gan, kps_eval)
        g_amp = evaluator.calculate_amplitude(df_gan, kps_eval)
        
        # Determinar Ganador (Basado en estabilidad: menor error óseo y jerk)
        # Normalizamos simple: quien tenga menor suma de (bone + jerk*100) gana estabilidad
        score_cvae = c_bone + (c_jerk * 100) 
        score_gan = g_bone + (g_jerk * 100)
        winner = "CVAE" if score_cvae < score_gan else "GAN"

        print(f"   📊 CVAE -> BoneErr: {c_bone:.4f} | Jerk: {c_jerk:.4f} | Amplitud: {c_amp:.4f}")
        print(f"   📊 GAN  -> BoneErr: {g_bone:.4f} | Jerk: {g_jerk:.4f} | Amplitud: {g_amp:.4f}")
        print(f"   🏆 Ganador estabilidad: {winner}")

        results_summary.append({
            'Escenario': escenario,
            'Modelo': 'CVAE',
            'Bone Error': c_bone,
            'Jerk': c_jerk,
            'Amplitud': c_amp
        })
        results_summary.append({
            'Escenario': escenario,
            'Modelo': 'GAN',
            'Bone Error': g_bone,
            'Jerk': g_jerk,
            'Amplitud': g_amp
        })

    # ==========================================
    # 5. VISUALIZACIÓN FINAL
    # ==========================================
    if not results_summary:
        print("No se generaron resultados.")
        return

    df_res = pd.DataFrame(results_summary)
    
    # Configurar Gráfica
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Gráfica 1: Bone Error
    sns.barplot(data=df_res, x='Escenario', y='Bone Error', hue='Modelo', ax=axes[0], palette="muted")
    axes[0].set_title('Consistencia Anatómica (Menor es mejor)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Gráfica 2: Jerk
    sns.barplot(data=df_res, x='Escenario', y='Jerk', hue='Modelo', ax=axes[1], palette="muted")
    axes[1].set_title('Suavidad / Jerk (Menor es mejor)')
    axes[1].tick_params(axis='x', rotation=45)

    # Gráfica 3: Amplitud
    sns.barplot(data=df_res, x='Escenario', y='Amplitud', hue='Modelo', ax=axes[2], palette="muted")
    axes[2].set_title('Dinamismo / Amplitud (Mayor es mejor)')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('comparacion_completa_json.png')
    print("\n" + "="*80)
    print("✓ Análisis completado. Gráfica guardada como 'comparacion_completa_json.png'")
    print("="*80)
    plt.show()

if __name__ == "__main__":
    ejecutar_comparacion_masiva()