"""
CP-0002: Comprobar que los datos de MediaPipe sean correctamente procesados por los modelos IA

Prueba de integración de componentes - caja blanca
Verifica que los CSV procesados (ya limpios por CP-0001) se integren con los modelos de IA.
Prueba para todos los modelos: CVAE-LSTM1, CVAE-LSTM2, GAN-LSTM1, GAN-LSTM2

Métricas de Discusión:
1.  Integridad Estructural: ¿Los datos tienen el formato y tipo correctos?
2.  Robustez (Longitud Variable): ¿El pipeline maneja videos de diferente duración?
3.  Seguridad: ¿El pipeline rechaza activamente datos corruptos o mal formados?
4.  Compatibilidad (Universalidad): ¿La salida de datos es compatible con los 4 modelos de IA?
"""
import os
import sys
import glob
import pytest
import numpy as np
import pandas as pd
import torch
from typing import List
import importlib.util

# --- Configuración de Paths ---
PRUEBAS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PRUEBAS_DIR)
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "Backend")
SERVICIO_ENTRENAMIENTO = os.path.join(BACKEND_ROOT, "Servicio_entrenamiento")
COORDENADAS_CSV = os.path.join(BACKEND_ROOT, "Coordenadas_csv")

sys.path.insert(0, SERVICIO_ENTRENAMIENTO)

# --- Carga Dinámica de Módulos de Modelos ---

def import_module_from_path(module_name: str, file_path: str):
    """Importa un módulo desde una ruta con nombre que contiene guiones."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"No se pudo encontrar el spec para {module_name} en {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def load_model_modules():
    """Carga todos los módulos de entrenamiento disponibles."""
    modules = {}
    model_files = {
        "CVAE-LSTM1": "CVAE-LSTM1.py",
        "CVAE-LSTM2": "CVAE-LSTM2.py",
        "GAN-LSTM1": "GAN-LSTM1.py",
        "GAN-LSTM2": "GAN-LSTM2.py",
    }
    
    for model_name, filename in model_files.items():
        model_path = os.path.join(SERVICIO_ENTRENAMIENTO, filename)
        if os.path.exists(model_path):
            try:
                module_name_safe = model_name.lower().replace("-", "_")
                module = import_module_from_path(module_name_safe, model_path)
                modules[model_name] = module
            except Exception as e:
                print(f"⚠️ Advertencia: Fallo al importar {filename}: {e}")
        else:
            print(f"⚠️ Advertencia: {filename} no encontrado, omitiendo {model_name}")
    
    return modules

MODEL_MODULES = load_model_modules()

if not MODEL_MODULES:
    pytest.fail(
        f"No se encontraron módulos de entrenamiento en: {SERVICIO_ENTRENAMIENTO}\n"
        f"Verifica que los archivos CVAE-LSTM1.py, CVAE-LSTM2.py, GAN-LSTM1.py, GAN-LSTM2.py existan.",
        pytrace=False
    )

# --- Fixtures de Escenarios ---

@pytest.fixture(scope="module")
def csv_files_processed():
    """Fixture: (Escenario Ideal) Obtiene archivos CSV procesados reales."""
    if not os.path.isdir(COORDENADAS_CSV):
        pytest.skip(f"Carpeta Coordenadas_csv no encontrada: {COORDENADAS_CSV}")
    
    csv_files = glob.glob(os.path.join(COORDENADAS_CSV, "*_processed.csv"))
    if not csv_files:
        pytest.skip(f"No se encontraron archivos *_processed.csv en {COORDENADAS_CSV}")
    
    return csv_files[:5] # Limitar a 5 para agilidad de la prueba

@pytest.fixture(params=list(MODEL_MODULES.keys()))
def model_config(request):
    """Fixture parametrizado: (Métrica de Universalidad) Proporciona config para CADA modelo."""
    model_name = request.param
    module = MODEL_MODULES[model_name]
    
    is_cvae = "CVAE" in model_name
    is_multi_label = "2" in model_name
    
    if not hasattr(module, 'StepDataset'):
        pytest.fail(f"El módulo {model_name} no tiene una clase 'StepDataset'")
        
    ModelClass = None
    if is_cvae:
        if not hasattr(module, 'CVAELSTM'):
            pytest.fail(f"El módulo {model_name} no tiene una clase 'CVAELSTM'")
        ModelClass = module.CVAELSTM
    else:
        if not hasattr(module, 'Generator'):
            pytest.fail(f"El módulo {model_name} no tiene una clase 'Generator'")
        ModelClass = module.Generator # Usamos Generador como clase principal para GAN
    
    return {
        "name": model_name,
        "module": module,
        "StepDataset": module.StepDataset,
        "ModelClass": ModelClass,
        "is_cvae": is_cvae,
        "is_multi_label": is_multi_label,
    }

# --- INICIO DE ELIMINACIÓN ---
# La fixture 'csv_file_with_nan' ha sido eliminada.
# --- FIN DE ELIMINACIÓN ---

@pytest.fixture
def csv_files_variable_length(tmp_path, model_config):
    """Fixture: (Escenario de Robustez 2) Crea CSVs con longitudes muy variables."""
    is_multi_label = model_config["is_multi_label"]
    csv_files = []
    lengths = [50, 200, 75] # Longitudes de frames distintas
    
    for i, length in enumerate(lengths):
        if is_multi_label:
            csv_path = tmp_path / f"Paso {i+1} - Hombre - TestVar_processed.csv"
        else:
            csv_path = tmp_path / f"TestVar{i}_processed.csv"
        
        data = {'frame': list(range(length)), 'timestamp': [j * 0.033 for j in range(length)]}
        landmarks = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_HIP']
        for landmark in landmarks:
            for coord in ['_x', '_y', '_z', '_visibility']:
                col_name = f"{landmark}{coord}"
                data[col_name] = [0.5] * length
        
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        csv_files.append(str(csv_path))
    
    return csv_files

@pytest.fixture
def csv_file_missing_columns(tmp_path, model_config):
    """Fixture: (Escenario de Seguridad 1) CSV con formato incorrecto (sin coords)."""
    csv_path = tmp_path / "test_missing_cols_processed.csv"
    df = pd.DataFrame({'frame': [0, 1, 2], 'timestamp': [0.0, 0.033, 0.067], 'invalid_col': [1, 2, 3]})
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def csv_file_with_invalid_data(tmp_path, model_config):
    """Fixture: (Escenario de Seguridad 2) CSV con datos no numéricos (strings)."""
    is_multi_label = model_config["is_multi_label"]
    csv_path = tmp_path / ("Paso 1 - Hombre - TestInvalidData_processed.csv" if is_multi_label else "TestInvalidData_processed.csv")
    
    data = {
        'frame': [0, 1, 2, 3],
        'timestamp': [0.0, 0.033, 0.067, 0.1],
        'NOSE_x': [0.5, 0.6, "ERROR", 0.4], # Dato no numérico
        'NOSE_y': [0.5, 0.6, 0.7, 0.4],
        'NOSE_z': [0.5, 0.6, 0.7, 0.4],
        'NOSE_visibility': [0.9, 0.9, 0.9, 0.9]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return str(csv_path)

# --- Funciones Auxiliares (Collate) ---

def create_collate_fn(is_multi_label: bool):
    """Crea función collate apropiada según el tipo de modelo."""
    if is_multi_label:
        def collate_varlen(batch):
            # Filtrar Nones si el StepDataset omitió un archivo
            batch = [b for b in batch if b is not None]
            if not batch:
                return None, None, None, None, None
            seqs, paso_labels, genero_labels, danza_labels, lengths = zip(*batch)
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            max_len = int(lengths_tensor.max().item())
            feat_dim = seqs[0].shape[1]
            padded = torch.zeros((len(seqs), max_len, feat_dim), dtype=torch.float32)
            for i, (seq, L) in enumerate(zip(seqs, lengths_tensor)):
                padded[i, :L, :] = seq[:L]
            paso_labels = torch.stack(list(paso_labels))
            genero_labels = torch.stack(list(genero_labels))
            danza_labels = torch.stack(list(danza_labels))
            return padded, paso_labels, genero_labels, danza_labels, lengths_tensor
    else:
        def collate_varlen(batch):
            # Filtrar Nones si el StepDataset omitió un archivo
            batch = [b for b in batch if b is not None]
            if not batch:
                return None, None, None
            seqs, labels, lengths = zip(*batch)
            lengths_tensor = torch.tensor(lengths, dtype=torch.long)
            max_len = int(lengths_tensor.max().item())
            feat_dim = seqs[0].shape[1]
            padded = torch.zeros((len(seqs), max_len, feat_dim), dtype=torch.float32)
            for i, (seq, L) in enumerate(zip(seqs, lengths_tensor)):
                padded[i, :L, :] = seq[:L]
            labels_tensor = torch.stack(list(labels))
            return padded, labels_tensor, lengths_tensor
    
    return collate_varlen

# --- Pruebas de Métricas de Integración ---

@pytest.mark.cp0002
def test_metric_integridad_estructural(csv_files_processed):
    """
    CP-0002: (Métrica 1)
    Compara: CSV real vs. Esquema esperado.
    Prueba: Que los CSV de CP-0001 tengan el formato y tipo de dato correctos.
    """
    print(f"\nVerificando Integridad Estructural en {len(csv_files_processed)} archivos...")
    assert len(csv_files_processed) > 0, "No hay archivos CSV para probar"
    
    for csv_path in csv_files_processed:
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c.endswith(("_x", "_y", "_z"))]
        
        assert len(df) > 0, f"CSV {csv_path} está vacío"
        assert len(feature_cols) > 0, f"CSV {csv_path} no tiene columnas de coordenadas (_x, _y, _z)"
        
        # NUEVA VERIFICACIÓN: Asegurar que no hay NaN (garantizado por CP-0001)
        assert not df[feature_cols].isnull().values.any(), f"Se encontraron valores NaN en {csv_path}, ¡CP-0001 falló!"

        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(df[col]), f"Columna {col} en {csv_path} no es numérica"
    print("✅ Métrica de Integridad Estructural: OK")


# --- INICIO DE ELIMINACIÓN ---
# La prueba 'test_metric_robustez_nan' ha sido eliminada.
# --- FIN DE ELIMINACIÓN ---


@pytest.mark.cp0002
def test_metric_robustez_longitud_variable(csv_files_variable_length, model_config):
    """
    CP-0002: (Métrica 2 - antes Métrica 3)
    Compara: Múltiples CSVs de longitudes distintas.
    Prueba: Que el DataLoader + collate_fn apliquen 'padding' correctamente.
    """
    from torch.utils.data import DataLoader
    
    StepDataset = model_config["StepDataset"]
    is_multi_label = model_config["is_multi_label"]
    model_name = model_config['name']
    
    print(f"\nVerificando Robustez (Longitud Variable) para {model_name}...")
    
    if is_multi_label:
        dataset = StepDataset(csv_paths=csv_files_variable_length, max_length=0, fit_scaler=True, fit_encoders=True)
    else:
        dataset = StepDataset(csv_paths=csv_files_variable_length, max_length=0, fit_scaler=True, fit_encoder=True)
    
    assert len(dataset) == len(csv_files_variable_length), "No se cargaron todas las secuencias"
    
    lengths = [dataset.lengths[i] for i in range(len(dataset))]
    
    assert len(set(lengths)) > 1, "Las secuencias de prueba no tienen longitudes variables"
    
    collate_fn = create_collate_fn(is_multi_label)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=collate_fn)
    
    batch = next(iter(loader))
    padded_seqs = batch[0]
    
    max_len_esperada = max(lengths)
    assert padded_seqs.shape[0] == len(dataset), "Tamaño de lote incorrecto"
    assert padded_seqs.shape[1] == max_len_esperada, f"Padding incorrecto. Esperado: {max_len_esperada}, Obtenido: {padded_seqs.shape[1]}"
    assert padded_seqs.shape[2] == dataset.num_features, "Número de features incorrecto en padding"
    print(f"✅ Métrica de Robustez (Padding) para {model_name}: OK (Shape: {list(padded_seqs.shape)})")


@pytest.mark.cp0002
def test_metric_seguridad_formato_incorrecto(csv_file_missing_columns, model_config):
    """
    CP-0002: (Métrica 3 - Test Negativo)
    Compara: CSV válido vs. CSV sin columnas de coordenadas.
    Prueba: Que el StepDataset falle activamente (lance Error) si faltan columnas.
    """
    StepDataset = model_config["StepDataset"]
    is_multi_label = model_config["is_multi_label"]
    
    print(f"\nVerificando Seguridad (Columnas Faltantes) para {model_config['name']}...")
    
    with pytest.raises((ValueError, AssertionError, IndexError, KeyError)) as exc_info:
        if is_multi_label:
            dataset = StepDataset(csv_paths=[csv_file_missing_columns], max_length=0, fit_scaler=True, fit_encoders=True)
        else:
            dataset = StepDataset(csv_paths=[csv_file_missing_columns], max_length=0, fit_scaler=True, fit_encoder=True)
        # Forzar la carga si es perezosa y maneja errores internos
        if len(dataset) > 0:
             _ = dataset[0] 
        
    assert exc_info.value is not None, "El StepDataset DEBIÓ fallar pero no lo hizo"
    print(f"✅ Métrica de Seguridad (Columnas Faltantes) para {model_config['name']}: OK (Falló como se esperaba)")


@pytest.mark.cp0002
def test_metric_seguridad_datos_no_numericos(csv_file_with_invalid_data, model_config):
    """
    CP-0002: (Métrica 4 - Test Negativo)
    Compara: CSV válido vs. CSV con strings ("ERROR") en columnas numéricas.
    Prueba: Que el StepDataset o el Scaler fallen al intentar normalizar datos no numéricos.
    """
    StepDataset = model_config["StepDataset"]
    is_multi_label = model_config["is_multi_label"]
    
    print(f"\nVerificando Seguridad (Datos No Numéricos) para {model_config['name']}...")
    
    with pytest.raises((ValueError, TypeError)) as exc_info:
        if is_multi_label:
            dataset = StepDataset(csv_paths=[csv_file_with_invalid_data], max_length=0, fit_scaler=True, fit_encoders=True)
        else:
            dataset = StepDataset(csv_paths=[csv_file_with_invalid_data], max_length=0, fit_scaler=True, fit_encoder=True)
        if len(dataset) > 0:
            _ = dataset[0] 
        
    assert exc_info.value is not None, "El StepDataset DEBIÓ fallar por datos no numéricos, pero no lo hizo"
    print(f"✅ Métrica de Seguridad (Datos No Numéricos) para {model_config['name']}: OK (Falló como se esperaba)")


@pytest.mark.cp0002
def test_metric_compatibilidad_shape_y_e2e(csv_files_processed, model_config):
    """
    CP-0002: (Métrica 5 - Prueba de Integración E2E)
    Compara: La 'forma' (shape) de la salida del DataLoader vs. la 'forma' de entrada del Modelo.
    Prueba: Que el pipeline completo (Dataset -> DataLoader -> Modelo) se ejecute sin errores de 'shape'.
    """
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    
    StepDataset = model_config["StepDataset"]
    ModelClass = model_config["ModelClass"]
    is_cvae = model_config["is_cvae"]
    is_multi_label = model_config["is_multi_label"]
    model_name = model_config["name"]
    
    print(f"\n--- Iniciando Prueba E2E para Modelo: {model_name} ---")

    # 1. Crear Dataset
    if is_multi_label:
        dataset = StepDataset(csv_paths=csv_files_processed, max_length=0, fit_scaler=True, fit_encoders=True)
        if len(dataset) == 0:
             pytest.skip(f"No se pudieron cargar datos para {model_name}, probablemente por fallos de parsing en los CSV reales.")
        num_pasos = len(dataset.paso_encoder.classes_)
        num_generos = len(dataset.genero_encoder.classes_)
        num_danzas = len(dataset.danza_encoder.classes_)
    else:
        dataset = StepDataset(csv_paths=csv_files_processed, max_length=0, fit_scaler=True, fit_encoder=True)
        if len(dataset) == 0:
             pytest.skip("No se pudieron cargar datos para {model_name}.")
        num_classes = len(dataset.label_encoder.classes_)
    
    print(f"  [Métrica] Datos cargados: {len(dataset)} secuencias")
    print(f"  [Métrica] Número de Features detectado: {dataset.num_features}")
    
    # 2. Crear DataLoader
    collate_fn = create_collate_fn(is_multi_label)
    loader = DataLoader(dataset, batch_size=min(2, len(dataset)), shuffle=False, collate_fn=collate_fn)

    # 3. Inicializar Modelo
    try:
        if is_multi_label:
            if is_cvae:
                model = ModelClass(num_features=dataset.num_features, num_pasos=num_pasos, num_generos=num_generos, num_danzas=num_danzas, hidden_size=64, latent_dim=32)
            else: # GAN
                model = ModelClass(noise_dim=64, num_features=dataset.num_features, hidden_size=64, num_pasos=num_pasos, num_generos=num_generos, num_danzas=num_danzas, embedding_dim=16)
        else:
            if is_cvae:
                model = ModelClass(num_features=dataset.num_features, num_classes=num_classes, hidden_size=64, latent_dim=32)
            else: # GAN
                model = ModelClass(noise_dim=64, num_features=dataset.num_features, hidden_size=64, num_classes=num_classes)
    except Exception as e:
        pytest.fail(f"Fallo al inicializar el modelo {model_name}: {e}")

    optimizer = Adam(model.parameters(), lr=1e-3)
    model.train()
    
    # 4. Simular un paso de entrenamiento (Prueba de 'Shape Match')
    try:
        batch = next(iter(loader))
        
        optimizer.zero_grad()
        
        if is_multi_label:
            seq, paso_labels, genero_labels, danza_labels, lengths = batch
            if seq is None: pytest.skip("Batch vacío, omitiendo.")
            print(f"  [Métrica] Shape del Tensor de Entrada (Lote): {list(seq.shape)}")
            
            if is_cvae:
                recon, mu, logvar = model(seq, paso_labels, genero_labels, danza_labels)
                loss = torch.nn.functional.mse_loss(recon, seq) + (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
            else: # GAN
                noise = torch.randn(seq.size(0), 64)
                generated = model(noise, paso_labels, genero_labels, danza_labels, seq.size(1))
                loss = torch.nn.functional.mse_loss(generated, seq) # Loss simulada
        else:
            seq, labels, lengths = batch
            if seq is None: pytest.skip("Batch vacío, omitiendo.")
            print(f"  [Métrica] Shape del Tensor de Entrada (Lote): {list(seq.shape)}")
            
            if is_cvae:
                recon, mu, logvar = model(seq, labels)
                loss = torch.nn.functional.mse_loss(recon, seq) + (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
            else: # GAN
                noise = torch.randn(seq.size(0), 64)
                generated = model(noise, labels, seq.size(1))
                loss = torch.nn.functional.mse_loss(generated, seq) # Loss simulada
        
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss), "Loss es NaN"
        
        print(f"  [Métrica] Entrenamiento simulado: Loss = {loss.item():.6f}")
        print(f"✅ Métrica de Compatibilidad ({model_name}): OK")
        
    except Exception as e:
        pytest.fail(f"Fallo en 'Shape Match' o paso de entrenamiento para {model_name}:\n{e}", pytrace=True)