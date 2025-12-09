import os
import math
import glob
import pandas as pd
import pytest
import numpy as np
import sys

# --- Definiciones Cinemáticas ---
LANDMARKS = {
    # Los ángulos se mantienen como un 'sanity check' (prueba de coherencia)
    "R_Knee_Angle": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
    "R_Hip_Angle": ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"]
}
EXPECTED_LANDMARKS = 32

# --- Funciones de Cálculo Cinemático ---

def calculate_angle_3d(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calcula el ángulo (en grados) entre 3 puntos 3D (p2 es el vértice)."""
    v21 = p1 - p2
    v23 = p3 - p2
    
    dot_product = np.sum(v21 * v23)
    magnitude_21 = np.linalg.norm(v21)
    magnitude_23 = np.linalg.norm(v23)
    
    if magnitude_21 == 0 or magnitude_23 == 0:
        return np.nan
        
    cosine_angle = dot_product / (magnitude_21 * magnitude_23)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle_rad = np.arccos(cosine_angle)
    return np.degrees(angle_rad)


def _calculate_quality_metrics(df: pd.DataFrame, angle_landmarks: dict) -> dict:
    """
    Calcula métricas de ESTABILIDAD (Aceleración) y COHERENCIA (Ángulos).
    """
    metrics = {}
    
    # 1. Cálculo de Ángulos (Prueba de Coherencia)
    for name, points in angle_landmarks.items():
        coords = {}
        for i, p_name in enumerate(points):
            cols_to_check = [f"{p_name}_x", f"{p_name}_y", f"{p_name}_z"]
            if not all(col in df.columns for col in cols_to_check):
                print(f"Advertencia: Faltan columnas para '{p_name}'. Saltando ángulo '{name}'.")
                coords = None
                break
            coords[f'p{i+1}'] = df[cols_to_check].to_numpy()
        
        if coords is None:
            continue

        angles = []
        for i in range(len(df)):
            p1 = coords['p1'][i]
            p2 = coords['p2'][i]
            p3 = coords['p3'][i]
            
            if not (np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any()):
                angles.append(calculate_angle_3d(p1, p2, p3))
            else:
                angles.append(np.nan)
        
        angles_series = pd.Series(angles, index=df.index)
        
        # Guardamos Max/Min para verificar que los ángulos son lógicos
        metrics[f"Max_{name}"] = angles_series.max()
        metrics[f"Min_{name}"] = angles_series.min()
        # --- RANGO ELIMINADO ---

    # 2. Métrica de Estabilidad (Aceleración)
    # Usamos RIGHT_WRIST como ejemplo para detectar 'jitter'
    r_wrist_cols = [c for c in df.columns if c.startswith("RIGHT_WRIST_") and c.endswith(("_x", "_y", "_z"))]
    
    if len(r_wrist_cols) == 3:
        coords_xyz = df[r_wrist_cols].to_numpy()
        diff = np.diff(coords_xyz, axis=0)
        distance = np.sqrt(np.sum(diff**2, axis=1))
        
        # --- DISTANCIA Y VELOCIDAD ELIMINADAS ---
        
        # 3. Aceleración (Métrica clave de estabilidad)
        dt = 1/30 # Asumimos 30 FPS
        velocity_change = np.diff(distance, axis=0)
        
        if velocity_change.size > 0:
            acceleration = velocity_change / dt
            metrics["Avg_Acceleration_R_Wrist"] = np.mean(np.abs(acceleration))
            metrics["Max_Acceleration_R_Wrist"] = np.max(np.abs(acceleration))
        else:
            metrics["Avg_Acceleration_R_Wrist"] = np.nan
            metrics["Max_Acceleration_R_Wrist"] = np.nan
    else:
        print("Advertencia: No se encontraron columnas de 'RIGHT_WRIST' para métricas de estabilidad.")
        metrics["Avg_Acceleration_R_Wrist"] = np.nan
        metrics["Max_Acceleration_R_Wrist"] = np.nan
        
    # --- DTW ELIMINADO ---
        
    return metrics


# --- El código principal de la prueba ---

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "Backend")
SERVICIO_EXTRACCION = os.path.join(BACKEND_ROOT, "Servicio_extraccion")


def _get_videos_for_test() -> list[str]:
    """
    Encuentra las rutas de video para la prueba parametrizada.
    
    Lógica:
    1. Si se define TEST_VIDEO_MP4, usa solo ese video.
    2. Si no, busca en 'BD/':
        - Añade el primer video encontrado (mp4 o mov).
        - Busca específicamente 'Paso 4 - Mujer - Turco' y lo añade si no es el mismo.
    """
    env_video = os.environ.get("TEST_VIDEO_MP4")
    if env_video and os.path.exists(env_video):
        # Opción 1: Variable de entorno tiene prioridad
        print(f"Usando video de variable de entorno: {env_video}")
        return [env_video]

    # Opción 2: Buscar en la carpeta BD
    bd_dir = os.path.join(os.path.dirname(PROJECT_ROOT), "BD")
    videos_to_test = []
    
    if not os.path.isdir(bd_dir):
        # Si no hay BD, no podemos encontrar nada
        print(f"Advertencia: No se encontró el directorio {bd_dir}")
        return []

    # Encontrar todos los videos candidatos
    candidates: list[str] = []
    for pattern in ("*.mp4", "*.mov"):
        candidates.extend(glob.glob(os.path.join(bd_dir, "**", pattern), recursive=True))

    if not candidates:
        # No se encontraron videos en BD
        return []

    # 1. Añadir el primer video encontrado
    first_video = candidates[0]
    videos_to_test.append(first_video)

    # 2. Buscar y añadir "Paso 4 - Mujer - Turco"
    paso_4_video_path = None
    for video in candidates:
        if "Paso 4 - Mujer - Turco" in os.path.basename(video):
            paso_4_video_path = video
            break
    
    if paso_4_video_path and paso_4_video_path not in videos_to_test:
        # Añadir solo si se encontró Y no es el mismo que el primero
        videos_to_test.append(paso_4_video_path)
    
    print(f"Videos encontrados para la prueba: {[os.path.basename(v) for v in videos_to_test]}")
    return videos_to_test

# --- Obtener la lista de videos ANTES de que Pytest recolecte las pruebas ---
VIDEO_LIST = _get_videos_for_test()


@pytest.mark.cp0001
@pytest.mark.parametrize("video_path", VIDEO_LIST)
def test_extraccion_pose_cp0001(video_path: str):

    # Si la lista de videos estaba vacía, la prueba debe saltarse.
    if not VIDEO_LIST:
         pytest.skip(f"No se encontró TEST_VIDEO_MP4 ni videos en {os.path.join(os.path.dirname(PROJECT_ROOT), 'BD')}")

    # --- La lógica de búsqueda de video se eliminó de aquí ---
    #     'video_path' ahora se recibe como parámetro.
    
    print(f"\n--- INICIANDO PRUEBA CP0001 PARA: {os.path.basename(video_path)} ---")

    assert os.path.exists(video_path), f"Video no encontrado: {video_path}"
    
    sys.path.insert(0, SERVICIO_EXTRACCION)
    try:
        from Pipeline import Pipeline
    except ImportError:
        pytest.fail(f"No se pudo importar 'Pipeline' desde {SERVICIO_EXTRACCION}. Verifica la ruta.")

    pipe = Pipeline(video_path=video_path, smooth_enabled=True, fix_legs=True)
    df_raw, df_processed = pipe.run()

    # Cambiamos el nombre de la función para reflejar el nuevo propósito
    quality_metrics = _calculate_quality_metrics(df_processed, LANDMARKS)
    movement_duration = len(df_processed)
    
    # Verificación 1: No vacíos
    assert len(df_raw) > 0, "No se detectó ninguna pose en el video (df_raw vacío)"
    assert len(df_processed) > 0, "Procesado vacío"

    # Verificación 2: Columnas esperadas
    coord_suffixes = ["_x", "_y", "_z", "_visibility"]
    coord_cols = [c for c in df_raw.columns if any(c.endswith(s) for s in coord_suffixes)]
    landmark_names_found = set(c.rsplit('_', 1)[0] for c in coord_cols)
    
    assert len(coord_cols) >= (EXPECTED_LANDMARKS * 4), (
        f"Faltan columnas de coordenadas/visibilidad. "
        f"Esperado: >={EXPECTED_LANDMARKS * 4}, Encontrado: {len(coord_cols)}"
    )
    assert len(landmark_names_found) >= EXPECTED_LANDMARKS, (
        f"No se encontraron suficientes landmarks únicos. "
        f"Esperado: >={EXPECTED_LANDMARKS}, Encontrado: {len(landmark_names_found)}"
    )


    # Verificación 3: Normalización de datos [0, 1]
    for coord in ("x", "y", "z"):
        cols = [c for c in df_processed.columns if c.endswith(f"_{coord}")]
        assert len(cols) > 0, f"No se encontraron columnas que terminen en _{coord}"
        
        vals = df_processed[cols].to_numpy()
        vals_valid = vals[~pd.isna(vals)]
        if len(vals_valid) > 0:
            assert (vals_valid >= 0).all() and (vals_valid <= 1).all(), \
                f"Valores fuera de [0,1] detectados en {coord}"


    # Verificación 4: Frames descartados (Check de calidad)
    assert len(df_processed) <= len(df_raw), (
        "El procesado no debe añadir frames. "
        f"Raw: {len(df_raw)}, Procesado: {len(df_processed)}"
    )
    
    # Verificación 5: Calidad de visibilidad (MÉTRICA CLAVE)
    vis_cols = [c for c in df_processed.columns if c.endswith("_visibility")]
    vis_mean = df_processed[vis_cols].mean().mean()
    # Esta es una métrica de calidad fundamental
    assert vis_mean >= 0.5, f"La visibilidad media es muy baja ({vis_mean:.3f})"
    
    quality_cols = [c for c in df_processed.columns if c.endswith("_quality")]
    assert len(quality_cols) >= EXPECTED_LANDMARKS, "No se generaron las columnas de '_quality'"


    # Verificación 6: CSV creado
    output_dir = os.path.dirname(video_path)
    try:
        test_csv_path = os.path.join(output_dir, "test_export_validation.csv")
        df_processed.to_csv(test_csv_path, index=False)
        assert os.path.exists(test_csv_path), "No se pudo crear archivo CSV de validación"
        os.remove(test_csv_path)
    except Exception as e:
        pytest.fail(f"Fallo en validación de exportación CSV: {str(e)}")


    # Verificación final: Impresión de Métricas
    print(f"\n✅ CP-0001 ÉXITO para {os.path.basename(video_path)}:")
    print(f"--- Métricas de Calidad de Extracción ---")
    print(f"   - Confianza de Detección (Visibilidad Media): {vis_mean:.4f}")
    print(f"   - Frames procesados (Duración): {movement_duration}")
    print(f"   - Frames descartados (Raw-Proc): {len(df_raw) - len(df_processed)}")
    print(f"   - CSV creado: ✓")

    print(f"\n--- Métricas de Estabilidad y Coherencia ---")

    # 1. Grupo de Ángulos (Coherencia)
    print(f"   📐 Coherencia de Ángulos (Valores Max/Min):")
    angle_keys = [k for k in quality_metrics if "Angle" in k]
    if angle_keys:
        for key in angle_keys:
            if isinstance(quality_metrics[key], float):
                print(f"     - {key}: {quality_metrics[key]:.4f}")
            else:
                print(f"     - {key}: {quality_metrics[key]}")
    else:
        print("     - No se calcularon métricas de ángulo.")

    # 2. Grupo de Aceleración (Estabilidad vs Jitter)
    print(f"   ⚡ Estabilidad de Extracción (Aceleración):")
    accel_keys = [k for k in quality_metrics if "Acceleration" in k]
    if accel_keys:
        for key in accel_keys:
            if isinstance(quality_metrics[key], float):
                print(f"     - {key}: {quality_metrics[key]:.6f}")
            else:
                print(f"     - {key}: {quality_metrics[key]}")
    else:
        print("     - No se calcularon métricas de aceleración.")

    # --- SECCIONES DE VELOCIDAD Y DTW ELIMINADAS ---