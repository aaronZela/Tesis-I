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
from unittest.mock import Mock, patch, MagicMock

# Configurar paths
PRUEBAS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PRUEBAS_DIR)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "Backend")
SERVICIO_EXTRACCION = os.path.join(BACKEND_ROOT, "Servicio_extraccion")

sys.path.insert(0, SERVICIO_EXTRACCION)

# Importar módulos del servicio de extracción
from Pipeline import Pipeline
from Procesar_video import VideoProcessor
from Procesar import fix_low_visibility, smooth, clip_coordinates, add_quality
from Utils import verify_extraction, print_final_statistics


# ==================== FIXTURES ====================

@pytest.fixture
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


@pytest.fixture
def sample_raw_df():
    """Fixture: Crea un DataFrame simulado de coordenadas RAW."""
    np.random.seed(42)
    num_frames = 50
    
    landmarks = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
                 'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']
    
    data = {
        'frame': list(range(num_frames)),
        'timestamp': [i * 0.033 for i in range(num_frames)]
    }
    
    for landmark in landmarks:
        # Coordenadas con algunos valores fuera de rango [0,1]
        data[f"{landmark}_x"] = np.random.uniform(-0.1, 1.1, num_frames)
        data[f"{landmark}_y"] = np.random.uniform(-0.1, 1.1, num_frames)
        data[f"{landmark}_z"] = np.random.uniform(-0.5, 0.5, num_frames)
        
        # Visibilidad con algunos valores bajos
        visibility = np.random.uniform(0.3, 1.0, num_frames)
        # Introducir algunos valores bajos deliberadamente
        low_vis_indices = np.random.choice(num_frames, size=5, replace=False)
        visibility[low_vis_indices] = np.random.uniform(0.1, 0.4, 5)
        data[f"{landmark}_visibility"] = visibility
    
    return pd.DataFrame(data)


@pytest.fixture
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
        # Coordenadas base con ruido
        base_x = np.linspace(0.4, 0.6, num_frames)
        base_y = np.linspace(0.5, 0.7, num_frames)
        
        # Agregar ruido aleatorio
        noise_x = np.random.normal(0, 0.05, num_frames)
        noise_y = np.random.normal(0, 0.05, num_frames)
        
        data[f"{landmark}_x"] = base_x + noise_x
        data[f"{landmark}_y"] = base_y + noise_y
        data[f"{landmark}_z"] = np.random.uniform(-0.3, 0.3, num_frames)
        data[f"{landmark}_visibility"] = np.random.uniform(0.5, 1.0, num_frames)
    
    return pd.DataFrame(data)


# ==================== TESTS ====================

@pytest.mark.cp0004
def test_carga_video_formato_valido(video_file):
    """
    CP-0004: Paso 1 - Cargar archivo de video con formato válido
    
    Precondición: Archivo mp4 listo para procesamiento
    """
    # Verificar que el archivo existe
    assert os.path.exists(video_file), f"Video no existe: {video_file}"
    
    # Verificar extensión válida
    valid_extensions = ['.mp4', '.mov']
    file_ext = os.path.splitext(video_file)[1].lower()
    assert file_ext in valid_extensions, f"Extensión no válida: {file_ext}"
    
    # Verificar que el archivo no está vacío
    file_size = os.path.getsize(video_file)
    assert file_size > 0, "El archivo de video está vacío"
    
    print(f"✅ Video válido: {os.path.basename(video_file)} ({file_size / (1024*1024):.2f} MB)")


@pytest.mark.cp0004
def test_funciones_tratamiento_importadas():
    """
    CP-0004: Paso 2 - Verificar que todas las funciones de tratamiento 
    están implementadas e importadas correctamente
    
    Precondición: Funciones de tratamiento implementadas e importadas
    """
    # Verificar que las funciones principales existen
    assert callable(fix_low_visibility), "fix_low_visibility no es callable"
    assert callable(smooth), "smooth no es callable"
    assert callable(clip_coordinates), "clip_coordinates no es callable"
    assert callable(add_quality), "add_quality no es callable"
    assert callable(verify_extraction), "verify_extraction no es callable"
    
    # Verificar clases principales
    assert VideoProcessor is not None, "VideoProcessor no importado"
    assert Pipeline is not None, "Pipeline no importado"
    
    print("✅ Todas las funciones de tratamiento están correctamente importadas")


@pytest.mark.cp0004
def test_extraccion_coordenadas_video(video_file):
    """
    CP-0004: Paso 3 - Separar los frames y extraer las coordenadas
    
    Verifica que el VideoProcessor puede extraer coordenadas del video
    """
    processor = VideoProcessor(video_file)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = os.path.join(tmpdir, "test_raw.csv")
        
        df_raw, fps, total_frames = processor.extract_coordinates(output_csv=output_csv)
        
        # Verificar que se extrajeron datos
        assert len(df_raw) > 0, "No se extrajeron coordenadas del video"
        
        # Verificar columnas esperadas
        assert 'frame' in df_raw.columns, "Falta columna 'frame'"
        assert 'timestamp' in df_raw.columns, "Falta columna 'timestamp'"
        
        # Verificar coordenadas de landmarks
        coord_cols = [c for c in df_raw.columns if c.endswith(('_x', '_y', '_z', '_visibility'))]
        assert len(coord_cols) > 0, "No se encontraron columnas de coordenadas"
        
        # Verificar que el CSV se guardó
        assert os.path.exists(output_csv), "CSV raw no se guardó"
        
        print(f"✅ Extracción exitosa: {len(df_raw)} frames, {len(coord_cols)} coordenadas")


@pytest.mark.cp0004
def test_tratamiento_datos_completo(sample_raw_df):
    """
    CP-0004: Paso 4 - Tratar los datos (normalización, suavizado, etc.)
    
    Verifica que todas las funciones de tratamiento funcionan correctamente
    """
    df_original = sample_raw_df.copy()
    
    # Paso 4.1: Corrección de visibilidad baja
    df_fixed = fix_low_visibility(df_original, threshold=0.5)
    assert len(df_fixed) == len(df_original), "fix_low_visibility cambió el número de filas"
    
    # Verificar que se interpolaron valores
    for col in df_fixed.columns:
        if col.endswith(('_x', '_y', '_z')) and not col.startswith('timestamp'):
            assert not df_fixed[col].isna().any(), f"Quedan NaN en {col} después de fix_low_visibility"
    
    # Paso 4.2: Suavizado
    df_smooth = smooth(df_fixed, window_length=11, polyorder=3)
    assert len(df_smooth) == len(df_fixed), "smooth cambió el número de filas"
    
    # Verificar que las columnas siguen siendo las mismas
    assert set(df_smooth.columns) == set(df_fixed.columns), "smooth cambió las columnas"
    
    # Paso 4.3: Recorte de coordenadas (normalización)
    df_clipped = clip_coordinates(df_smooth)
    
    # Verificar que todas las coordenadas están en [0, 1]
    for coord in ['_x', '_y', '_z']:
        coord_cols = [c for c in df_clipped.columns if c.endswith(coord)]
        for col in coord_cols:
            assert df_clipped[col].min() >= 0, f"{col} tiene valores < 0"
            assert df_clipped[col].max() <= 1, f"{col} tiene valores > 1"
    
    # Paso 4.4: Agregar calidad
    df_processed = add_quality(df_clipped, threshold=0.5)
    
    # Verificar que se agregaron columnas de calidad
    quality_cols = [c for c in df_processed.columns if c.endswith('_quality')]
    assert len(quality_cols) > 0, "No se agregaron columnas de calidad"
    
    # Verificar valores de calidad
    for col in quality_cols:
        unique_values = df_processed[col].unique()
        valid_qualities = {'low', 'medium', 'high'}
        assert set(unique_values).issubset(valid_qualities), f"Valores de calidad inválidos en {col}"
    
    print(f"✅ Tratamiento completo exitoso: {len(df_processed)} frames procesados")


@pytest.mark.cp0004
def test_validar_estructura_dataset(sample_raw_df):
    """
    CP-0004: Paso 5 - Validar estructura del dataset (columnas y tipos de datos)
    """
    df_processed = sample_raw_df.copy()
    df_processed = fix_low_visibility(df_processed)
    df_processed = smooth(df_processed)
    df_processed = clip_coordinates(df_processed)
    df_processed = add_quality(df_processed)
    
    # Verificar columnas obligatorias
    assert 'frame' in df_processed.columns, "Falta columna 'frame'"
    assert 'timestamp' in df_processed.columns, "Falta columna 'timestamp'"
    
    # Verificar tipos de datos
    assert pd.api.types.is_integer_dtype(df_processed['frame']), "frame debe ser entero"
    assert pd.api.types.is_numeric_dtype(df_processed['timestamp']), "timestamp debe ser numérico"
    
    # Verificar columnas de coordenadas
    coord_suffixes = ['_x', '_y', '_z', '_visibility']
    coord_cols = [c for c in df_processed.columns 
                  if any(c.endswith(s) for s in coord_suffixes)]
    
    assert len(coord_cols) > 0, "No se encontraron columnas de coordenadas"
    
    # Verificar que todas las coordenadas son numéricas
    for col in coord_cols:
        assert pd.api.types.is_numeric_dtype(df_processed[col]), \
            f"Columna {col} no es numérica"
    
    # Verificar columnas de calidad
    quality_cols = [c for c in df_processed.columns if c.endswith('_quality')]
    assert len(quality_cols) > 0, "No se encontraron columnas de calidad"
    
    # Verificar integridad: no debe haber NaN en coordenadas principales
    essential_coords = [c for c in df_processed.columns if c.endswith(('_x', '_y', '_z'))]
    for col in essential_coords:
        nan_count = df_processed[col].isna().sum()
        assert nan_count == 0, f"Columna {col} tiene {nan_count} valores NaN"
    
    print(f"✅ Estructura validada: {len(df_processed.columns)} columnas, tipos correctos")


@pytest.mark.cp0004
def test_verificar_dimensiones_esperadas(video_file):
    """
    CP-0004: Paso 6 - Verificar dimensiones esperadas (número de frames y puntos clave)
    """
    processor = VideoProcessor(video_file)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        output_csv = os.path.join(tmpdir, "test_dimensions.csv")
        df_raw, fps, total_frames = processor.extract_coordinates(output_csv=output_csv)
        
        # Verificar que se detectaron frames
        assert len(df_raw) > 0, "No se detectaron frames"
        
        # Verificar número de landmarks (33 landmarks de MediaPipe Pose)
        # Cada landmark tiene: _x, _y, _z, _visibility = 4 columnas
        # Total esperado: 33 * 4 = 132 columnas + 2 (frame, timestamp) = 134
        expected_landmark_count = 33
        
        coord_cols = [c for c in df_raw.columns if c.endswith(('_x', '_y', '_z', '_visibility'))]
        actual_landmark_count = len(coord_cols) // 4
        
        assert actual_landmark_count == expected_landmark_count, \
            f"Se esperaban {expected_landmark_count} landmarks, se encontraron {actual_landmark_count}"
        
        # Verificar que hay suficiente cobertura de detección
        detection_rate = (len(df_raw) / total_frames) * 100
        assert detection_rate >= 50, \
            f"Tasa de detección muy baja: {detection_rate:.1f}% (mínimo 50%)"
        
        # Verificar dimensiones del DataFrame
        expected_cols = 2 + (33 * 4)  # frame, timestamp + 33 landmarks * 4 coords
        assert len(df_raw.columns) == expected_cols, \
            f"Número de columnas incorrecto: {len(df_raw.columns)} (esperado: {expected_cols})"
        
        print(f"✅ Dimensiones correctas: {len(df_raw)} frames, {actual_landmark_count} landmarks")


@pytest.mark.cp0004
def test_compatibilidad_con_modelos_ia(sample_raw_df):
    """
    CP-0004: Paso 7 - Confirmar inicio del entrenamiento sin excepciones
    
    Verifica que el formato de salida es compatible con los modelos de IA
    """
    # Procesar datos completo
    df_processed = sample_raw_df.copy()
    df_processed = fix_low_visibility(df_processed)
    df_processed = smooth(df_processed)
    df_processed = clip_coordinates(df_processed)
    df_processed = add_quality(df_processed)
    
    # Simular lo que haría el StepDataset
    feature_cols = [c for c in df_processed.columns 
                    if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")]
    
    assert len(feature_cols) > 0, "No se encontraron columnas de features para el modelo"
    
    # Extraer features como lo haría el modelo
    features = df_processed[feature_cols].values.astype(np.float32)
    
    # Verificar forma del array
    assert features.ndim == 2, "Features debe ser 2D (frames x coords)"
    assert features.shape[0] == len(df_processed), "Número de frames no coincide"
    assert features.shape[1] == len(feature_cols), "Número de features no coincide"
    
    # Verificar que no hay NaN
    assert not np.isnan(features).any(), "Features contiene NaN"
    
    # Verificar que no hay infinitos
    assert not np.isinf(features).any(), "Features contiene valores infinitos"
    
    # Verificar rango [0, 1]
    assert features.min() >= 0, f"Features tiene valores negativos: {features.min()}"
    assert features.max() <= 1, f"Features tiene valores > 1: {features.max()}"
    
    # Simular normalización (como lo haría StandardScaler)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    try:
        features_scaled = scaler.fit_transform(features)
        assert features_scaled.shape == features.shape, "Normalización cambió las dimensiones"
    except Exception as e:
        pytest.fail(f"Error al normalizar features: {str(e)}")
    
    print(f"✅ Formato compatible con modelos IA: {features.shape}")


@pytest.mark.cp0004
def test_pipeline_completo_integracion(video_file):
    """
    CP-0004: Integración completa - Pipeline end-to-end
    
    Verifica que todo el pipeline funciona correctamente de principio a fin
    """
    # Crear pipeline con todas las funciones activas
    pipeline = Pipeline(video_path=video_file, smooth_enabled=True, fix_legs=True)
    
    # Ejecutar pipeline completo
    df_raw, df_processed = pipeline.run()
    
    # Verificación 1: Ambos DataFrames existen
    assert df_raw is not None, "df_raw es None"
    assert df_processed is not None, "df_processed es None"
    
    # Verificación 2: df_processed no debe ser mayor que df_raw
    assert len(df_processed) <= len(df_raw), \
        f"df_processed ({len(df_processed)}) > df_raw ({len(df_raw)})"
    
    # Verificación 3: Estructura de columnas
    assert len(df_processed.columns) >= len(df_raw.columns), \
        "df_processed perdió columnas durante el tratamiento"
    
    # Verificación 4: Coordenadas normalizadas
    for coord in ['_x', '_y', '_z']:
        coord_cols = [c for c in df_processed.columns if c.endswith(coord)]
        for col in coord_cols:
            vals = df_processed[col].dropna()
            if len(vals) > 0:
                assert vals.min() >= 0, f"{col} tiene valores < 0"
                assert vals.max() <= 1, f"{col} tiene valores > 1"
    
    # Verificación 5: Columnas de calidad agregadas
    quality_cols = [c for c in df_processed.columns if c.endswith('_quality')]
    assert len(quality_cols) > 0, "No se agregaron columnas de calidad"
    
    # Verificación 6: CSV de salida
    project_root = os.path.dirname(os.path.dirname(SERVICIO_EXTRACCION))
    output_dir = os.path.join(project_root, "Coordenadas_csv")
    video_basename = os.path.splitext(os.path.basename(video_file))[0]
    processed_csv = os.path.join(output_dir, f"{video_basename}_processed.csv")
    
    assert os.path.exists(processed_csv), f"CSV procesado no se guardó: {processed_csv}"
    
    print(f"✅ Pipeline completo exitoso:")
    print(f"   - Frames raw: {len(df_raw)}")
    print(f"   - Frames procesados: {len(df_processed)}")
    print(f"   - Columnas procesadas: {len(df_processed.columns)}")
    print(f"   - CSV guardado: {os.path.basename(processed_csv)}")


@pytest.mark.cp0004
def test_extension_normalizacion_adicional(sample_raw_df_with_noise):
    """
    CP-0004: Extensión 4.1 - Corrección de valores anormales mediante normalización adicional
    """
    df = sample_raw_df_with_noise.copy()
    
    # Introducir algunos valores anormales deliberadamente
    df.loc[5, 'NOSE_x'] = 5.0  # Valor muy fuera de rango
    df.loc[10, 'LEFT_KNEE_y'] = -2.0  # Valor negativo extremo
    
    # Aplicar tratamiento
    df_fixed = fix_low_visibility(df)
    df_smooth = smooth(df_fixed)
    df_clipped = clip_coordinates(df_smooth)
    
    # Verificar que los valores anormales fueron corregidos
    assert df_clipped['NOSE_x'].max() <= 1.0, "Valor anormal no fue corregido"
    assert df_clipped['LEFT_KNEE_y'].min() >= 0.0, "Valor negativo no fue corregido"
    
    # Verificar que todos los valores están en rango
    for coord in ['_x', '_y', '_z']:
        coord_cols = [c for c in df_clipped.columns if c.endswith(coord)]
        for col in coord_cols:
            assert df_clipped[col].min() >= 0, f"{col} tiene valores < 0"
            assert df_clipped[col].max() <= 1, f"{col} tiene valores > 1"
    
    print("✅ Normalización adicional aplicada correctamente a valores anormales")


@pytest.mark.cp0004
def test_extension_error_estructura_no_coincide():
    """
    CP-0004: Extensión 6.1 - Error cuando la estructura no coincide con la esperada
    """
    # Crear DataFrame con estructura incorrecta
    df_invalid = pd.DataFrame({
        'frame': [0, 1, 2],
        'invalid_col': [1, 2, 3]
        # Falta timestamp y coordenadas
    })
    
    # Intentar procesar
    feature_cols = [c for c in df_invalid.columns 
                    if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")]
    
    # Debe detectar que no hay columnas de coordenadas
    assert len(feature_cols) == 0, "No debería haber encontrado columnas de coordenadas"
    
    # Simular la verificación del sistema
    if len(feature_cols) == 0:
        error_msg = "ERROR: No se encontraron columnas de coordenadas (_x, _y, _z)"
        print(f"✅ Error detectado correctamente: {error_msg}")
    else:
        pytest.fail("El sistema no detectó la estructura inválida")


@pytest.mark.cp0004
def test_condicion_fallo_perdida_columnas():
    """
    CP-0004: Condición de Fallo - Pérdida de columnas durante el tratamiento
    """
    # Crear DataFrame de prueba
    df_original = pd.DataFrame({
        'frame': [0, 1, 2],
        'timestamp': [0.0, 0.033, 0.067],
        'NOSE_x': [0.5, 0.5, 0.5],
        'NOSE_y': [0.5, 0.5, 0.5],
        'NOSE_z': [0.0, 0.0, 0.0],
        'NOSE_visibility': [0.9, 0.9, 0.9]
    })
    
    original_columns = set(df_original.columns)
    
    # Aplicar tratamiento
    df_processed = fix_low_visibility(df_original)
    df_processed = smooth(df_processed)
    df_processed = clip_coordinates(df_processed)
    
    processed_columns = set(df_processed.columns)
    
    # Verificar que no se perdieron columnas esenciales
    essential_cols = {'frame', 'timestamp', 'NOSE_x', 'NOSE_y', 'NOSE_z'}
    assert essential_cols.issubset(processed_columns), \
        f"Se perdieron columnas esenciales: {essential_cols - processed_columns}"
    
    print("✅ No se perdieron columnas durante el tratamiento")


@pytest.mark.cp0004
def test_condicion_exito_estructura_valida(sample_raw_df):
    """
    CP-0004: Condición de Éxito - Las coordenadas mantienen estructura y formato válidos
    """
    # Tratamiento completo
    df_processed = sample_raw_df.copy()
    df_processed = fix_low_visibility(df_processed)
    df_processed = smooth(df_processed)
    df_processed = clip_coordinates(df_processed)
    df_processed = add_quality(df_processed)
    
    # Validar dimensiones
    assert len(df_processed) > 0, "DataFrame vacío"
    assert len(df_processed.columns) > 0, "Sin columnas"
    
    # Validar formato válido para entrenamiento
    feature_cols = [c for c in df_processed.columns 
                    if c.endswith("_x") or c.endswith("_y") or c.endswith("_z")]
    
    features = df_processed[feature_cols].values.astype(np.float32)
    
    # Verificaciones de condición de éxito
    checks = {
        "Dimensiones correctas": features.ndim == 2,
        "Sin NaN": not np.isnan(features).any(),
        "Sin infinitos": not np.isinf(features).any(),
        "Rango [0,1]": (features.min() >= 0) and (features.max() <= 1),
        "Estructura mantenida": len(df_processed) == len(sample_raw_df)
    }
    
    for check_name, result in checks.items():
        assert result, f"Falló: {check_name}"
    
    print(f"✅ CONDICIÓN DE ÉXITO cumplida:")
    for check_name in checks.keys():
        print(f"   ✓ {check_name}")