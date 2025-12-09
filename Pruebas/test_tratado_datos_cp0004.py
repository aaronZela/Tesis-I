"""
CP-0004: Comprobar que el proceso interno de tratado de datos mantenga la estructura
esperada y los valores dentro del rango definido antes del uso de los modelos IA.

Prueba de servicio de extracción - caja blanca
Verifica el tratamiento correcto de datos antes de entrenar los modelos.
"""
import os
import sys
import glob
import pytest
import numpy as np
import pandas as pd
import tempfile

# Configurar paths
PRUEBAS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PRUEBAS_DIR)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "Backend")
SERVICIO_EXTRACCION = os.path.join(BACKEND_ROOT, "Servicio_extraccion")

sys.path.insert(0, SERVICIO_EXTRACCION)

# Importar módulos del servicio de extracción
try:
    from Pipeline import Pipeline
    from Procesar_video import VideoProcessor
    from Procesar import fix_low_visibility, smooth, clip_coordinates, add_quality
    from Utils import verify_extraction, print_final_statistics
except ImportError as e:
    pytest.fail(f"Fallo al importar módulos del Servicio_extraccion: {e}")


# ==================== HELPERS ====================

def _calculate_jitter(df: pd.DataFrame, col: str) -> float:
    """Calcula el 'jitter' (aceleración media) de una columna."""
    # Jitter se mide como la media de la segunda derivada (aceleración)
    # diff().diff() calcula la segunda diferencia
    accel = df[col].diff().diff().abs()
    return accel.mean()

# ==================== FIXTURES ====================

@pytest.fixture(scope="module")
def video_file():
    """Fixture: Busca un video válido para pruebas."""
    video_path = os.environ.get("TEST_VIDEO_MP4")
    
    if not video_path or not os.path.exists(video_path):
        bd_dir = os.path.join(PROJECT_ROOT, "BD")
        candidates = []
        
        if os.path.isdir(bd_dir):
            for pattern in ("*.mp4", "*.mov"):
                candidates.extend(glob.glob(os.path.join(bd_dir, "**", pattern), recursive=True))
        
        if candidates:
            video_path = candidates[0]
        else:
            pytest.skip(f"No se encontró video de prueba en {bd_dir}")
    
    assert os.path.exists(video_path), f"Video no encontrado: {video_path}"
    return video_path


@pytest.fixture(scope="module")
def sample_raw_df_with_noise():
    """Fixture: DataFrame RAW con ruido y valores extremos."""
    np.random.seed(123)
    num_frames = 30
    
    landmarks = ['NOSE', 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    
    data = {
        'frame': list(range(num_frames)),
        'timestamp': [i * 0.033 for i in range(num_frames)]
    }
    
    for landmark in landmarks:
        base_x = np.linspace(0.4, 0.6, num_frames)
        base_y = np.linspace(0.5, 0.7, num_frames)
        noise_x = np.random.normal(0, 0.05, num_frames)
        noise_y = np.random.normal(0, 0.05, num_frames)
        
        data[f"{landmark}_x"] = base_x + noise_x
        data[f"{landmark}_y"] = base_y + noise_y
        data[f"{landmark}_z"] = np.random.uniform(-0.3, 0.3, num_frames)
        data[f"{landmark}_visibility"] = np.random.uniform(0.5, 1.0, num_frames)
    
    # Introducir valores anormales
    data['NOSE_x'][5] = 5.0
    data['LEFT_KNEE_y'][10] = -2.0
    # Introducir NaN
    data['RIGHT_KNEE_x'][15:20] = np.nan
        
    return pd.DataFrame(data)


# ==================== TESTS DE FUNCIONES ====================
# Estas pruebas usan datos simulados (fixtures) para probar
# que las funciones de limpieza responden correctamente a
# datos sucios (NaN, ruido, valores anormales).

@pytest.mark.cp0004
def test_funciones_tratamiento_importadas():
    """Paso 2: Verificar que todas las funciones de tratamiento existen"""
    assert callable(fix_low_visibility), "fix_low_visibility no es callable"
    assert callable(smooth), "smooth no es callable"
    assert callable(clip_coordinates), "clip_coordinates no es callable"
    assert callable(add_quality), "add_quality no es callable"
    print("✅ Todas las funciones de tratamiento están correctamente importadas")


@pytest.mark.cp0004
def test_limpieza_datos_sucios(sample_raw_df_with_noise):
    """
    CP-0004: Paso 4 - Tratar los datos (normalización, suavizado, etc.)
    Prueba unitaria con datos simulados "sucios".
    """
    df_sucio = sample_raw_df_with_noise.copy()
    
    # --- Métrica 1: Conteo de NaN (Interpolación) ---
    print("\n--- [Prueba Unitaria: Robustez a NaN] ---")
    coord_cols_xyz = [c for c in df_sucio.columns if c.endswith(('_x', '_y', '_z'))]
    nan_count_antes = df_sucio[coord_cols_xyz].isna().sum().sum()
    print(f"Valores NaN (Antes): {nan_count_antes}")
    assert nan_count_antes > 0, "El set de datos de prueba 'sample_raw_df_with_noise' no contenía NaN."
    
    df_fixed = fix_low_visibility(df_sucio) # 'fix_low_visibility' también interpola
    nan_count_despues = df_fixed[coord_cols_xyz].isna().sum().sum()
    assert nan_count_despues == 0, "Quedan NaN después de fix_low_visibility"
    print(f"Valores NaN (Después): {nan_count_despues}")
    print("✅ (Prueba Unitaria) fix_low_visibility manejó NaN correctamente.")

    # --- Métrica 2: Rango de Valores (Normalización/Clipping) ---
    print("\n--- [Prueba Unitaria: Robustez a Valores Anormales] ---")
    max_val_antes = df_fixed[coord_cols_xyz].max().max()
    min_val_antes = df_fixed[coord_cols_xyz].min().min()
    print(f"Rango de valores (Antes de clip): [{min_val_antes:.4f}, {max_val_antes:.4f}]")
    assert max_val_antes > 1.0 or min_val_antes < 0.0, "El set de datos de prueba no tenía valores anormales."

    df_clipped = clip_coordinates(df_fixed)
    max_val_despues = df_clipped[coord_cols_xyz].max().max()
    min_val_despues = df_clipped[coord_cols_xyz].min().min()
    assert max_val_despues <= 1.0 and min_val_despues >= 0.0, "clip_coordinates falló"
    print(f"Rango de valores (Después de clip): [{min_val_despues:.4f}, {max_val_despues:.4f}]")
    print("✅ (Prueba Unitaria) clip_coordinates manejó valores anormales correctamente.")

    # --- Métrica 3: Efectividad del Suavizado (Jitter) ---
    print("\n--- [Prueba Unitaria: Robustez a Ruido (Jitter)] ---")
    col_test = 'LEFT_KNEE_x'
    jitter_antes = _calculate_jitter(df_fixed, col_test) # Usamos df_fixed (ya interpolado)
    assert jitter_antes > 0.01, f"El Jitter 'antes' ({jitter_antes}) es muy bajo, la prueba de ruido no es válida"
    
    df_smooth = smooth(df_fixed)
    jitter_despues = _calculate_jitter(df_smooth, col_test)
    
    assert jitter_despues < jitter_antes, "El suavizado falló (Jitter no se redujo)"
    reduction_pct = ((jitter_antes - jitter_despues) / jitter_antes) * 100
    
    print(f"   - Jitter (Ruido) Antes: {jitter_antes:.6f}")
    print(f"   - Jitter (Ruido) Después: {jitter_despues:.6f}")
    print(f"✅ (Prueba Unitaria) smooth redujo el ruido en {reduction_pct:.2f}%")
    


# ==========================================================
# --- REPORTE FINAL DE MÉTRICAS (PRUEBA E2E CON VIDEO REAL) ---
# ==========================================================

@pytest.mark.cp0004
def test_reporte_metricas_video_real(video_file):
    """
    CP-0004: REPORTE FINAL DE MÉTRICAS (Pipeline E2E)
    
    Prueba el pipeline completo de Extracción (Paso 1-7) en un VIDEO REAL
    y genera el reporte final de métricas comparando ANTES (df_raw) y 
    DESPUÉS (df_processed).
    """
    print("\n" + "="*60)
    print("--- REPORTE FINAL DE MÉTRICAS (CP-0004) ---")
    print(f"--- Video Real: {os.path.basename(video_file)} ---")
    print("="*60)

    # 1. Ejecutar el Pipeline completo
    pipeline = Pipeline(video_path=video_file, smooth_enabled=True, fix_legs=True)
    df_raw, df_processed = pipeline.run()

    # 2. Obtener métricas de extracción (Tasa de Detección)
    processor = VideoProcessor(video_file)
    _, fps, total_frames = processor.extract_coordinates()
    
    print("\n--- [Métrica 1: Calidad de Extracción (MediaPipe)] ---")
    
    # Métrica: Tasa de Detección
    detection_rate = (len(df_raw) / total_frames) * 100
    assert detection_rate >= 50, f"Tasa de detección muy baja: {detection_rate:.1f}%"
    print(f"   - Tasa de Detección: {detection_rate:.1f}% ({len(df_raw)} / {total_frames} frames)")

    # Métrica: Conteo de Landmarks (33 de MediaPipe Pose)
    expected_landmark_count = 33
    coord_cols_raw = [c for c in df_raw.columns if c.endswith(('_x', '_y', '_z', '_visibility'))]
    actual_landmark_count = len(coord_cols_raw) // 4
    assert actual_landmark_count == expected_landmark_count, f"Se esperaban {expected_landmark_count} landmarks, se encontraron {actual_landmark_count}"
    print(f"   - Conteo de Landmarks: {actual_landmark_count} (Esperado: {expected_landmark_count})")
    
    # 3. Métricas de Procesamiento (Antes vs. Después)
    
    coord_cols_xyz_raw = [c for c in df_raw.columns if c.endswith(('_x', '_y', '_z'))]
    coord_cols_xyz_proc = [c for c in df_processed.columns if c.endswith(('_x', '_y', '_z'))]
    
    # --- Métrica 2: Efectividad de Interpolación (NaN) ---
    print("\n--- [Métrica 2: Efectividad de Interpolación (fix_low_visibility)] ---")
    nan_count_antes = df_raw[coord_cols_xyz_raw].isna().sum().sum()
    print(f"   - Conteo de NaN (Antes, en df_raw): {nan_count_antes}")
    
    nan_count_despues = df_processed[coord_cols_xyz_proc].isna().sum().sum()
    assert nan_count_despues == 0, "Quedan NaN después del procesamiento"
    print(f"   - Conteo de NaN (Después, en df_processed): {nan_count_despues}")
    print("✅ Métrica de Interpolación: OK (0 NaN en salida)")

    # --- Métrica 3: Efectividad de Normalización (Clipping) ---
    print("\n--- [Métrica 3: Efectividad de Normalización (clip_coordinates)] ---")
    max_val_antes = df_raw[coord_cols_xyz_raw].max().max()
    min_val_antes = df_raw[coord_cols_xyz_raw].min().min()
    print(f"   - Rango de Valores (Antes, en df_raw): [{min_val_antes:.4f}, {max_val_antes:.4f}]")

    max_val_despues = df_processed[coord_cols_xyz_proc].max().max()
    min_val_despues = df_processed[coord_cols_xyz_proc].min().min()
    assert max_val_despues <= 1.0 and min_val_despues >= 0.0, "clip_coordinates falló en el video real"
    print(f"   - Rango de Valores (Después, en df_processed): [{min_val_despues:.4f}, {max_val_despues:.4f}]")
    print("✅ Métrica de Normalización: OK (Rango [0, 1] verificado)")

    # --- Métrica 4: Efectividad del Suavizado (Jitter) ---
    print("\n--- [Métrica 4: Efectividad del Suavizado (smooth)] ---")
    col_test = 'LEFT_KNEE_x' # Columna de ejemplo
    
    jitter_antes = _calculate_jitter(df_raw, col_test)
    jitter_despues = _calculate_jitter(df_processed, col_test)
    
    # Es posible que el video real ya sea suave, así que solo reportamos
    assert jitter_despues <= jitter_antes, \
        f"El suavizado falló: Jitter 'después' ({jitter_despues}) es mayor que 'antes' ({jitter_antes})"
    
    reduction_pct = 0.0
    if jitter_antes > 0:
        reduction_pct = ((jitter_antes - jitter_despues) / jitter_antes) * 100
    
    print(f"   - Jitter (Ruido) Antes (df_raw): {jitter_antes:.6f}")
    print(f"   - Jitter (Ruido) Después (df_processed): {jitter_despues:.6f}")
    print(f"✅ Métrica de Suavizado: OK (Reducción del {reduction_pct:.2f}%)")

    # --- Métrica 5: Integridad Estructural (Compatibilidad IA) ---
    print("\n--- [Métrica 5: Integridad Estructural (Compatibilidad IA)] ---")
    feature_cols = [c for c in df_processed.columns if c.endswith(("_x", "_y", "_z"))]
    features = df_processed[feature_cols].values.astype(np.float32)
    
    checks = {
        "Columnas de Calidad añadidas": len(df_processed.columns) > len(df_raw.columns),
        "Shape de Features (frames, coords)": features.ndim == 2,
        "Sin Infinitos en Features": not np.isinf(features).any(),
    }
    
    print(f"   - Shape Final para IA: {features.shape}")
    
    for check_name, result in checks.items():
        assert result, f"Falló: {check_name}"
        print(f"   - {check_name}: OK")
    
    print("\n" + "="*60)
    print("✅ CONDICIÓN DE ÉXITO (CP-0004) CUMPLIDA")
    print("="*60)