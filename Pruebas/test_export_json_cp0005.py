"""
Código de Prueba CP-0005
Objetivo: Verificar exportación correcta de JSON sin pérdida de información
Alcance: Servicio de renderizado - Prueba de caja negra
Herramienta: PyTest
"""

import pytest
import json
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import time  # <-- 1. IMPORTACIÓN AÑADIDA

# Importar funciones de generación
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Backend.Servicio_IA.generate_cvae_lstm2 import generate_cvae_sample
from Backend.Servicio_IA.generate_gan_lstm2 import generate_gan_sample

# Definir rutas absolutas para los modelos
script_dir = Path(__file__).parent
backend_dir = script_dir.parent / 'Backend' / 'Servicio_IA_Entrenada'
CVAE_MODEL_PATH = str(backend_dir / 'cvae_lstm_best.pt')
GAN_MODEL_PATH = str(backend_dir / 'gan_lstm_best_model.pt')

# Nombres esperados por MediaPipe (para Métrica de Compatibilidad)
EXPECTED_MEDIAPIPE_NAMES = [
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

# --- 2. FUNCIÓN HELPER AÑADIDA ---
def _calculate_jitter(series: pd.Series) -> float:
    """Calcula el 'jitter' (aceleración media) de una serie."""
    # Jitter se mide como la media de la segunda derivada (aceleración)
    accel = series.diff().diff().abs()
    return accel.mean()
# --- FIN DE LA FUNCIÓN HELPER ---


class TestCP0005ExportacionJSON:
    """Suite de pruebas para CP-0005: Exportación JSON"""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Configuración y limpieza para cada prueba"""
        # Usar directorio temporal para pruebas en lugar de Danza_Nueva
        self.output_dir = Path(__file__).parent / 'test_temp_output'
        self.output_dir.mkdir(exist_ok=True)
        
        # Archivos generados durante las pruebas
        self.generated_files = []
        
        yield # <-- Aquí se ejecuta la prueba
        
        # --- Limpieza (Teardown) ---
        print("\nLimpiando archivos generados...")
        for f_path_str in self.generated_files:
            f_path = Path(f_path_str)
            if f_path.exists():
                try:
                    f_path.unlink()
                    print(f" - Eliminado: {f_path.name}")
                except OSError as e:
                    print(f" - ERROR al eliminar {f_path.name}: {e}")
    
    def test_precondicion_modelos_disponibles(self):
        """
        Precondición: Verificar que los modelos de IA están disponibles
        """
        cvae_model = Path(CVAE_MODEL_PATH)
        gan_model = Path(GAN_MODEL_PATH)

        assert cvae_model.exists(), "Modelo CVAE no encontrado"
        assert gan_model.exists(), "Modelo GAN no encontrado"
        print("✅ Precondición: Modelos CVAE y GAN encontrados.")

    @pytest.mark.parametrize("modelo,funcion_gen", [
        ("CVAE", generate_cvae_sample),
        ("GAN", generate_gan_sample)
    ])
    def test_generacion_y_seleccion_formato(self, modelo, funcion_gen):
        """
        Paso 1-2: Sistema genera secuencia y selecciona formato JSON
        """
        model_paths = {"CVAE": CVAE_MODEL_PATH, "GAN": GAN_MODEL_PATH}

        df, json_data = funcion_gen(
            model_path=model_paths[modelo],
            seq_length=30,
            paso=1,
            genero='Mujer',
            danza='Carnaval',
            fps=30
        )
        
        # Guardar JSON retornado para pruebas (ya no se guarda automáticamente)
        json_path = self.output_dir / f'generated_{modelo.lower()}_paso1_Mujer_Carnaval.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        self.generated_files.append(str(json_path))

        assert isinstance(df, pd.DataFrame), "Salida no es DataFrame"
        assert isinstance(json_data, dict), "Salida no es diccionario JSON"
        assert 'metadata' in json_data, "JSON sin metadata"
        assert 'frames' in json_data, "JSON sin frames"
        print(f"✅ {modelo} generó DataFrame y JSON correctamente.")
    
    # ========================================================================
    # --- PRUEBAS DE MÉTRICAS (AÑADIDAS Y MEJORADAS) ---
    # ========================================================================

    @pytest.mark.parametrize("modelo,funcion_gen", [
        ("CVAE", generate_cvae_sample),
        ("GAN", generate_gan_sample)
    ])
    def test_metric_integridad_datos_df_vs_json(self, modelo, funcion_gen):
        """
        CP-0005: (Métrica 1) - Pérdida de Datos (DataFrame vs. JSON)
        Compara: DataFrame en memoria vs. JSON guardado.
        Prueba: Que no hay pérdida de precisión numérica.
        """
        print(f"\n--- [Métrica 1: Pérdida de Datos (DataFrame vs. JSON) - {modelo}] ---")
        model_paths = {"CVAE": CVAE_MODEL_PATH, "GAN": GAN_MODEL_PATH}
        
        # 1. Generar datos
        df_original, json_original = funcion_gen(
            model_path=model_paths[modelo],
            seq_length=20,
            paso=1,
            genero='Hombre',
            danza='Carnaval',
            fps=30
        )
        
        # Guardar JSON retornado para pruebas (ya no se guarda automáticamente)
        json_path = self.output_dir / f'generated_{modelo.lower()}_paso1_Hombre_Carnaval.json'
        with open(json_path, 'w') as f:
            json.dump(json_original, f, indent=2)
        self.generated_files.append(str(json_path))
        
        # 2. Leer datos del JSON guardado
        assert json_path.exists(), "El archivo JSON no fue escrito"
        with open(json_path, 'r') as f:
            json_leido = json.load(f)

        # 3. Comparar un punto de muestra (ej. frame 10, keypoint 15)
        frame_idx = 10
        kp_idx = 15 # (RIGHT_ELBOW)
        
        # Datos del DataFrame original
        kp_name_df = EXPECTED_MEDIAPIPE_NAMES[kp_idx]
        df_row = df_original[df_original['frame'] == frame_idx]
        val_x_df = df_row[f'{kp_name_df}_x'].values[0]
        val_y_df = df_row[f'{kp_name_df}_y'].values[0]
        val_z_df = df_row[f'{kp_name_df}_z'].values[0]

        # Datos del JSON leído
        json_frame = json_leido['frames'][frame_idx]
        json_kp = json_frame['keypoints'][kp_idx]
        val_x_json = json_kp['x']
        val_y_json = json_kp['y']
        val_z_json = json_kp['z']

        # 4. Calcular Métrica (Error Absoluto)
        error_x = abs(val_x_df - val_x_json)
        error_y = abs(val_y_df - val_y_json)
        error_z = abs(val_z_df - val_z_json)
        error_total = error_x + error_y + error_z
        
        print(f"   - Comparando Frame {frame_idx}, Keypoint '{kp_name_df}'")
        print(f"   - DF (Original): (x={val_x_df:.6f}, y={val_y_df:.6f}, z={val_z_df:.6f})")
        print(f"   - JSON (Leído):   (x={val_x_json:.6f}, y={val_y_json:.6f}, z={val_z_json:.6f})")
        print(f"   - [Métrica] Error Absoluto Total: {error_total:.8f}")

        assert error_total < 1e-6, "Pérdida de precisión detectada"
        print(f"✅ Métrica de Integridad de Datos: OK (Sin pérdida de datos)")

    @pytest.mark.parametrize("modelo,funcion_gen", [
        ("CVAE", generate_cvae_sample),
        ("GAN", generate_gan_sample)
    ])
    def test_metric_estabilidad_numerica(self, modelo, funcion_gen):
        """
        CP-0005: (Métrica 2) - Estabilidad Numérica
        Prueba: Que el JSON no contiene NaN/Inf y la visibilidad está en [0, 1].
        """
        print(f"\n--- [Métrica 2: Estabilidad Numérica - {modelo}] ---")
        model_paths = {"CVAE": CVAE_MODEL_PATH, "GAN": GAN_MODEL_PATH}
        
        df, json_data = funcion_gen(
            model_path=model_paths[modelo],
            seq_length=10,
            paso=1,
            genero='Mujer',
            danza='Carnaval',
            fps=30
        )
        
        # Guardar JSON retornado para pruebas (ya no se guarda automáticamente)
        json_path = self.output_dir / f'generated_{modelo.lower()}_paso1_Mujer_Carnaval.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        self.generated_files.append(str(json_path))

        nan_count = 0
        inf_count = 0
        vis_out_of_range = 0

        for frame in json_data['frames']:
            for kp in frame['keypoints']:
                if not np.isfinite(kp['x']) or not np.isfinite(kp['y']) or not np.isfinite(kp['z']):
                    if np.isnan(kp['x']): nan_count += 1
                    if np.isinf(kp['x']): inf_count += 1
                if not (0 <= kp['visibility'] <= 1):
                    vis_out_of_range += 1
        
        print(f"   - [Métrica] Conteo de NaN: {nan_count}")
        print(f"   - [Métrica] Conteo de Infinitos: {inf_count}")
        print(f"   - [Métrica] Visibilidad fuera de Rango [0,1]: {vis_out_of_range}")

        assert nan_count == 0, "Valores NaN encontrados"
        assert inf_count == 0, "Valores Infinitos encontrados"
        assert vis_out_of_range == 0, "Valores de visibilidad fuera de rango [0,1]"
        print(f"✅ Métrica de Estabilidad Numérica: OK")

    @pytest.mark.parametrize("modelo,funcion_gen", [
        ("CVAE", generate_cvae_sample),
        ("GAN", generate_gan_sample)
    ])
    def test_metric_compatibilidad_blender(self, modelo, funcion_gen):
        """
        CP-0005: (Métrica 3) - Integridad Estructural y Compatibilidad Blender
        Prueba: Que la estructura (33 keypoints, nombres) coincide con la especificación.
        """
        print(f"\n--- [Métrica 3: Compatibilidad Blender - {modelo}] ---")
        model_paths = {"CVAE": CVAE_MODEL_PATH, "GAN": GAN_MODEL_PATH}
        
        df, json_data = funcion_gen(
            model_path=model_paths[modelo],
            seq_length=5,
            paso=1,
            genero='Hombre',
            danza='Carnaval',
            fps=30
        )
        
        # Guardar JSON retornado para pruebas (ya no se guarda automáticamente)
        json_path = self.output_dir / f'generated_{modelo.lower()}_paso1_Hombre_Carnaval.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        self.generated_files.append(str(json_path))

        frame = json_data['frames'][0]
        keypoints = frame['keypoints']
        
        # Métrica: Conteo de Keypoints
        kp_count = len(keypoints)
        print(f"   - [Métrica] Conteo de Keypoints: {kp_count} (Esperado: 33)")
        assert kp_count == 33, f"Se esperaban 33 keypoints, se encontraron {kp_count}"

        # Métrica: Coincidencia de Nombres
        actual_names = [kp['name'] for kp in keypoints]
        assert actual_names == EXPECTED_MEDIAPIPE_NAMES, "Nombres de keypoints no coinciden con MediaPipe"
        print(f"   - [Métrica] Coincidencia de Nombres: OK")

        # Métrica: Secuencialidad
        timestamps = [f['timestamp'] for f in json_data['frames']]
        is_sequential = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
        assert is_sequential, "Timestamps no son incrementales"
        print(f"   - [Métrica] Timestamps Secuenciales: OK")
        print(f"✅ Métrica de Compatibilidad Blender: OK")

    @pytest.mark.parametrize("modelo,funcion_gen", [
        ("CVAE", generate_cvae_sample),
        ("GAN", generate_gan_sample)
    ])
    def test_metric_escalabilidad_secuencia_larga(self, modelo, funcion_gen):
        """
        CP-0005: (Métrica 4) - Escalabilidad Estructural
        Prueba: Que la estructura del JSON es válida para secuencias largas.
        """
        print(f"\n--- [Métrica 4: Escalabilidad Estructural - {modelo}] ---")
        model_paths = {"CVAE": CVAE_MODEL_PATH, "GAN": GAN_MODEL_PATH}
        largo = 500 # 500 frames
        
        df, json_data = funcion_gen(
            model_path=model_paths[modelo],
            seq_length=largo,
            paso=1,
            genero='Hombre',
            danza='Carnaval',
            fps=30
        )
        
        # Guardar JSON retornado para pruebas (ya no se guarda automáticamente)
        json_path = self.output_dir / f'generated_{modelo.lower()}_paso1_Hombre_Carnaval.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        self.generated_files.append(str(json_path))

        # Métrica: Longitud de Frames
        frame_count = len(json_data['frames'])
        print(f"   - [Métrica] Longitud de Frames: {frame_count} (Esperado: {largo})")
        assert frame_count == largo, f"Se esperaban {largo} frames, se obtuvieron {frame_count}"

        # Métrica: Validez de Estructura (último frame)
        last_frame = json_data['frames'][-1]
        assert last_frame['frame'] == largo - 1, "Frame final no es secuencial"
        assert len(last_frame['keypoints']) == 33, "Último frame no tiene 33 keypoints"
        print(f"   - [Métrica] Estructura del último frame: OK")
        print(f"✅ Métrica de Escalabilidad: OK")
    

    # --- 3. NUEVAS MÉTRICAS AÑADIDAS DENTRO DE LA CLASE ---
    
    @pytest.mark.parametrize("modelo,funcion_gen", [
        ("CVAE", generate_cvae_sample),
        ("GAN", generate_gan_sample)
    ])
    def test_metric_calidad_movimiento_generado(self, modelo, funcion_gen):
        """
        CP-0005: (Métrica 5) - Calidad de Movimiento (Jitter/Vibración)
        Prueba: Que el movimiento generado no tenga vibraciones excesivas.
        """
        print(f"\n--- [Métrica 5: Calidad de Movimiento (Jitter) - {modelo}] ---")
        model_paths = {"CVAE": CVAE_MODEL_PATH, "GAN": GAN_MODEL_PATH}
        
        df, json_data = funcion_gen(
            model_path=model_paths[modelo],
            seq_length=100, # Usar una secuencia más larga para medir jitter
            paso=1,
            genero='Hombre',
            danza='Carnaval',
            fps=30
        )
        
        # Guardar JSON retornado para pruebas (ya no se guarda automáticamente)
        json_path = self.output_dir / f'generated_{modelo.lower()}_paso1_Hombre_Carnaval.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        self.generated_files.append(str(json_path))

        # Calcular jitter en un keypoint clave (ej. Muñeca Izquierda)
        jitter_x = _calculate_jitter(df['LEFT_WRIST_x'])
        jitter_y = _calculate_jitter(df['LEFT_WRIST_y'])
        jitter_z = _calculate_jitter(df['LEFT_WRIST_z'])
        
        avg_jitter = (jitter_x + jitter_y + jitter_z) / 3
        
        # Definir un umbral de vibración aceptable (puedes ajustar esto)
        MAX_JITTER_ACEPTABLE = 0.05 
        
        print(f"   - [Métrica] Jitter Promedio (LEFT_WRIST): {avg_jitter:.8f}")
        
        assert avg_jitter < MAX_JITTER_ACEPTABLE, f"Jitter excesivo detectado ({avg_jitter:.8f})"
        assert avg_jitter > 1e-6, "Jitter es cero, la animación puede estar atascada (stuck)"
        
        print(f"✅ Métrica de Calidad de Movimiento: OK (Jitter bajo)")

    @pytest.mark.parametrize("modelo,funcion_gen", [
        ("CVAE", generate_cvae_sample),
        ("GAN", generate_gan_sample)
    ])
    def test_metric_rendimiento_generacion(self, modelo, funcion_gen):
        """
        CP-0005: (Métrica 6) - Rendimiento de Generación
        Prueba: Que la generación y guardado se completa en un tiempo aceptable.
        """
        print(f"\n--- [Métrica 6: Rendimiento de Generación - {modelo}] ---")
        model_paths = {"CVAE": CVAE_MODEL_PATH, "GAN": GAN_MODEL_PATH}
        
        seq_len = 100
        limite_tiempo_seg = 5.0 # Límite: 5 segundos para 100 frames

        start_time = time.perf_counter()
        
        df, json_data = funcion_gen(
            model_path=model_paths[modelo],
            seq_length=seq_len,
            paso=1,
            genero='Mujer',
            danza='Carnaval',
            fps=30
        )
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Guardar JSON retornado para pruebas (ya no se guarda automáticamente)
        json_path = self.output_dir / f'generated_{modelo.lower()}_paso1_Mujer_Carnaval.json'
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        self.generated_files.append(str(json_path))
        
        file_size_kb = json_path.stat().st_size / 1024

        print(f"   - [Métrica] Tiempo de Generación ({seq_len} frames): {duration:.4f}s")
        print(f"   - [Métrica] Tamaño de Archivo: {file_size_kb:.2f} KB")

        assert duration < limite_tiempo_seg, f"Generación demasiado lenta ({duration:.4f}s)"
        assert file_size_kb > 10, "Tamaño de archivo sospechosamente pequeño"

        print(f"✅ Métrica de Rendimiento: OK")


    # ========================================================================
    # --- PRUEBAS DE VALIDACIÓN Y CONDICIÓN DE ÉXITO ---
    # ========================================================================

    def test_extension_formato_invalido_frames_incompletos(self):
        """
        Extensión 3.1: Detectar frames con estructura incompleta (Prueba Negativa)
        """
        json_invalido = {
            "metadata": {"paso": 1},
            "frames": [ { "frame": 0 } ] # Falta timestamp y keypoints
        }

        def validar_estructura_frame(frame):
            required = ['frame', 'timestamp', 'keypoints']
            for field in required:
                if field not in frame:
                    return False, f"Falta campo {field}"
            return True, "Válido"

        valido, mensaje = validar_estructura_frame(json_invalido['frames'][0])
        assert not valido, "Debería detectar frame incompleto"
        print(f"\n✅ Prueba Negativa (Frame Incompleto) OK: {mensaje}")

    @pytest.mark.parametrize("modelo,paso,genero,danza", [
        ("CVAE", 1, "Hombre", "Carnaval"),
        ("GAN", 3, "Mujer", "Carnaval"),
    ])
    def test_condicion_exito_flujo_completo(self, modelo, paso, genero, danza):
        """
        Condición de Éxito: Verificar flujo completo (Generar -> Validar -> Escribir -> Leer)
        """
        print(f"\n--- [Prueba Condición de Éxito: Flujo Completo {modelo}] ---")
        if modelo == "CVAE":
            funcion_gen = generate_cvae_sample
            model_path = CVAE_MODEL_PATH
            prefix = 'generated_cvae'
        else:
            funcion_gen = generate_gan_sample
            model_path = GAN_MODEL_PATH
            prefix = 'generated_gan'

        # 1. Generar secuencia
        df, json_data_original = funcion_gen(
            model_path=model_path,
            seq_length=20,
            paso=paso,
            genero=genero,
            danza=danza,
            fps=30
        )

        # 2. Guardar JSON retornado para pruebas (ya no se guarda automáticamente)
        json_filename = f"{prefix}_paso{paso}_{genero}_{danza}.json"
        json_path = self.output_dir / json_filename
        with open(json_path, 'w') as f:
            json.dump(json_data_original, f, indent=2)
        self.generated_files.append(str(json_path))
        
        assert json_path.exists(), f"Archivo JSON no creado: {json_path}"
        assert json_path.stat().st_size > 0, "Archivo JSON está vacío"
        print("   - Archivo escrito en disco: OK")

        # 3. Leer y Validar JSON
        try:
            with open(json_path, 'r') as f:
                json_data_leido = json.load(f)
        except json.JSONDecodeError as e:
            pytest.fail(f"Archivo JSON corrupto: {e}")
        
        assert json_data_leido == json_data_original, "JSON leído difiere del original"
        print("   - Lectura y validación de JSON: OK")

        # 4. Verificar compatibilidad con Blender
        assert len(json_data_leido['frames'][0]['keypoints']) == 33, "No tiene 33 keypoints"
        actual_names = [kp['name'] for kp in json_data_leido['frames'][0]['keypoints']]
        assert actual_names == EXPECTED_MEDIAPIPE_NAMES, "Nombres de keypoints no coinciden"
        print("   - Compatibilidad con Blender (33 keypoints, nombres): OK")
        print(f"✅ Condición de Éxito ({modelo}): OK")


# --- Funciones de Simulación (fuera de la clase) ---

def validar_json_para_blender(json_path: str) -> Tuple[bool, str]:
    """Valida que un archivo JSON es compatible con Blender API"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        return False, f"Error al leer JSON: {e}"

    if 'metadata' not in data: return False, "Falta metadata"
    if 'frames' not in data: return False, "Falta frames"
    if len(data['frames']) == 0: return False, "No hay frames"

    for i, frame in enumerate(data['frames']):
        if 'frame' not in frame: return False, f"Frame {i} sin número"
        if 'timestamp' not in frame: return False, f"Frame {i} sin timestamp"
        if 'keypoints' not in frame: return False, f"Frame {i} sin keypoints"
        if len(frame['keypoints']) != 33:
            return False, f"Frame {i} no tiene 33 keypoints (tiene {len(frame['keypoints'])})"
    return True, "JSON válido para Blender"


def simular_importacion_blender(json_path: str) -> bool:
    """Simula la importación del JSON en Blender"""
    valido, mensaje = validar_json_para_blender(json_path)
    if not valido:
        print(f"❌ Error en importación Blender: {mensaje}")
        return False
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"✓ Simulación Blender:")
    print(f" Standard metric: Frames a animar: {len(data['frames'])}")
    print(f" Standard metric: FPS: {data['metadata'].get('fps', 30)}")
    print(f" Standard metric: Duración: {len(data['frames']) / data['metadata'].get('fps', 30):.2f}s")
    print(f" Standard metric: Keypoints por frame: {len(data['frames'][0]['keypoints'])}")
    return True

# --- Función de Reporte Final ---
def generar_reporte_final():
    """Genera un mensaje de éxito después de la ejecución de PyTest."""
    print("\n" + "="*80)
    print(" Condición de Éxito CP-0005 Verificada con PyTest ")
    print("="*80)
    print("Todas las pruebas de exportación JSON pasaron exitosamente.")
    print("   - Se verificó la **integridad y el formato** del JSON (metadata, frames, 33 keypoints, tipos de datos).")
    print("   - Se validó la **Pérdida de Datos**: Error numérico < 1e-6.")
    print("   - Se validó la **Estabilidad Numérica**: 0 NaN, 0 Inf, Visibilidad en [0,1].")
    print("   - Se validó la **Compatibilidad con Blender**: Nombres de keypoints 100% coincidentes.")
    print("   - Se validó la **Escalabilidad**: Estructura válida con 500 frames.")
    
    # --- MENSAJES DE REPORTE PARA NUEVAS MÉTRICAS ---
    print("   - Se validó la **Calidad de Movimiento**: Jitter (vibración) dentro de límites aceptables.")
    print("   - Se validó el **Rendimiento**: Tiempo de generación y tamaño de archivo dentro de los límites.")
    
    print("\n Se puede proceder a la integración del servicio de renderizado.")
    print("="*80)
    
    try:
        # Usar directorio temporal de pruebas en lugar de Danza_Nueva
        output_dir_path = Path(__file__).parent / 'test_temp_output'
        test_file = output_dir_path / 'generated_cvae_paso1_Hombre_Carnaval.json'
        
        if test_file.exists():
            simular_importacion_blender(str(test_file))
        else:
            print(f"Aviso: Archivo de ejemplo no encontrado en {test_file}. El reporte de Blender no se pudo ejecutar.")
    except Exception as e:
        print(f"Error al intentar la simulación de Blender: {e}")

if __name__ == "__main__":
    # Ejecutar PyTest
    result = pytest.main([__file__, "-v", "--tb=short", "--color=yes"])
    
    # Generar el reporte final después de la ejecución de PyTest
    if result == 0:
        generar_reporte_final()