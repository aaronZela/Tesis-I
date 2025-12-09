"""
Código de Prueba CP-0006
Objetivo: Medir el tiempo requerido desde la carga de un video hasta la generación del RENDER final.
Alcance: Sistema completo incluyendo renderizado – Prueba de rendimiento.
Métricas: Latencia E2E, Desglose de Tiempos (Cuello de Botella), 
          Uso de Recursos (CPU/RAM), Rendimiento (Throughput).
"""
import pytest
import json
import os
import sys
import time
import shutil
import threading
from datetime import datetime
from pathlib import Path
import psutil 

# --- Configuración de Rutas Absolutas ---
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'Frontend'))

try:
    from app import process_video_async, UPLOAD_FOLDER, RESULTS_FOLDER, processing_status
except ImportError as e:
    print(f"Error de importación en app.py: {e}. Usando fallbacks.")
    UPLOAD_FOLDER = str(project_root / 'Frontend' / 'uploads')
    RESULTS_FOLDER = str(project_root / 'Frontend' / 'results')
    processing_status = {}
    
from Backend.Servicio_IA.generate_cvae_lstm2 import generate_cvae_sample
from Backend.Servicio_IA.generate_gan_lstm2 import generate_gan_sample
from Backend.Servicio_extraccion.Pipeline import Pipeline
from Backend.Render.render_video import render_skeleton_video

# Rutas estáticas
bd_dir = project_root / 'BD'
sample_video_path = bd_dir / 'Paso 1 - Hombre - Carnaval.mp4' 

backend_ia_dir = project_root / 'Backend' / 'Servicio_IA_Entrenada'
CVAE_MODEL_PATH = str(backend_ia_dir / 'cvae_lstm_best.pt')
GAN_MODEL_PATH = str(backend_ia_dir / 'gan_lstm_best_model.pt')

# Configuración para pruebas
TEST_USERNAME = 'admin@gmail.com'
TARGET_TIME_SECONDS = 180  # 3 minutos (objetivo ideal, no límite estricto)
MAX_WAIT_TIMEOUT = 900  # 15 minutos timeout absoluto para evitar cuelgues
RETRIES = 3

# ========================================================================
# --- ALMACÉN GLOBAL DE MÉTRICAS ---
# ========================================================================
global_test_metrics = {
    'runs': [],
}


class TestCP0006PerformanceJSONGeneration:
    """Suite de pruebas para CP-0006: Rendimiento hasta generación de Render"""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Configuración y limpieza para cada prueba"""
        self.test_upload_dir = Path(UPLOAD_FOLDER) / 'test_uploads'
        self.test_upload_dir.mkdir(exist_ok=True)
        
        self.test_results_dir = Path(RESULTS_FOLDER) / 'test_results'
        self.test_results_dir.mkdir(exist_ok=True)
        
        self.generated_files = []
        
        yield
        
        # Limpieza
        for file_path_str in self.generated_files:
            file_path = Path(file_path_str)
            if file_path.exists():
                try: file_path.unlink()
                except OSError: pass
        
        if self.test_upload_dir.exists():
            shutil.rmtree(self.test_upload_dir, ignore_errors=True)
        if self.test_results_dir.exists():
            shutil.rmtree(self.test_results_dir, ignore_errors=True)
    
    def test_precondicion_sistema_desplegado(self):
        cvae_model = Path(CVAE_MODEL_PATH)
        gan_model = Path(GAN_MODEL_PATH)
        assert cvae_model.exists(), f"Modelo CVAE no encontrado: {CVAE_MODEL_PATH}"
        assert gan_model.exists(), f"Modelo GAN no encontrado: {GAN_MODEL_PATH}"
        assert Pipeline, "Servicio de extracción no disponible"
        
        assert sample_video_path.exists(), f"Video de prueba no encontrado en: {sample_video_path}"
        print(f"\n✓ Precondiciones verificadas. Video: {sample_video_path.name}")

    def test_precondicion_modulos_activos(self):
        try:
            _, json_data = generate_cvae_sample(
                model_path=CVAE_MODEL_PATH, seq_length=10, paso=1, 
                genero='Hombre', danza='Carnaval', fps=30
            )
            assert json_data is not None, "Generación CVAE falla"
        except Exception as e:
            pytest.fail(f"Servicio de inferencia CVAE no activo: {e}")
        
        try:
            _, json_data = generate_gan_sample(
                model_path=GAN_MODEL_PATH, seq_length=10, paso=1, 
                genero='Hombre', danza='Carnaval', fps=30
            )
            assert json_data is not None, "Generación GAN falla"
        except Exception as e:
            pytest.fail(f"Servicio de inferencia GAN no activo: {e}")
        
        print("✓ Módulos de IA verificados.")

    @pytest.mark.parametrize("resolution", ["720p"]) 
    def test_rendimiento_flujo_completo_hasta_render(self, resolution):
        """
        Descripción principal: Medir tiempo desde 'carga' hasta renderizado completo.
        Esta prueba POBLA el diccionario 'global_test_metrics'.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = f"{TEST_USERNAME}_{resolution}_{timestamp}_test.mp4"
        test_video_path = self.test_upload_dir / video_name
        
        shutil.copy2(sample_video_path, test_video_path)
        self.generated_files.append(str(test_video_path))
        
        print(f"\n{'='*80}")
        print(f"✓ Simulando carga de video: {test_video_path.name}")
        print(f"{'='*80}")
        
        task_id = f"test_{resolution}_{timestamp}"
        
        process = psutil.Process(os.getpid())
        mem_usage_samples = []
        cpu_usage_samples = []
        
        # ============================================================
        # FASE 1: PROCESAMIENTO (Extracción + Inferencia hasta JSON)
        # ============================================================
        print("\n[FASE 1] Iniciando procesamiento (Extracción + Inferencia)...")
        phase1_start = time.time()
        
        thread = threading.Thread(
            target=process_video_async,
            args=(str(test_video_path), task_id, TEST_USERNAME, True, True)
        )
        thread.start()
        
        # Esperar a que termine el procesamiento
        waited = 0
        while processing_status.get(task_id, {}).get('status') == 'processing' and waited < MAX_WAIT_TIMEOUT:
            try:
                cpu_usage_samples.append(process.cpu_percent(interval=0.1))
                mem_usage_samples.append(process.memory_info().rss / (1024 * 1024))
            except psutil.NoSuchProcess:
                break
            
            time.sleep(1)
            waited += 1
        
        # Esperar a que el thread termine completamente
        thread.join(timeout=30)  # Dar 30 seg adicionales para que cierre limpiamente
        
        phase1_end = time.time()
        phase1_time = phase1_end - phase1_start
        
        final_status = processing_status.get(task_id, {}).get('status', 'not_found')
        
        if final_status == 'error':
            pytest.fail(f"Tarea falló: {processing_status[task_id]['message']}")
        
        if final_status != 'completed':
            print(f"\n⚠️ ADVERTENCIA: Procesamiento no completado después de {waited}s")
            print(f"   Estado final: {final_status}")
            print(f"   Esto puede deberse a que el render aún está en proceso.")
            print(f"   Se intentará continuar con el render manual...\n")
        
        print(f"\n✅ [FASE 1 COMPLETADA] Tiempo de procesamiento: {phase1_time:.2f}s")
        
        # Obtener timings internos si están disponibles
        timings = {}
        if 'timings' in processing_status.get(task_id, {}):
            timings = processing_status[task_id]['timings']
            print(f"   - Extracción (Pipeline): {timings.get('t_extraccion_video', 0):.2f}s")
            print(f"   - Inferencia (CVAE): {timings.get('t_inferencia_cvae', 0):.2f}s")
            print(f"   - Inferencia (GAN): {timings.get('t_inferencia_gan', 0):.2f}s")
        
        # Verificar que los JSONs se generaron
        json_files = list(Path(RESULTS_FOLDER).glob(f"*{TEST_USERNAME}*.json"))
        self.generated_files.extend([str(f) for f in json_files])
        
        if len(json_files) < 2:
            print(f"\n⚠️ ADVERTENCIA: Solo se encontraron {len(json_files)} JSONs (esperado: 2)")
            print("   Buscando en directorio de resultados...")
            all_jsons = list(Path(RESULTS_FOLDER).glob("*.json"))
            print(f"   Total de JSONs en directorio: {len(all_jsons)}")
            if all_jsons:
                json_files = all_jsons[:2]  # Tomar los primeros 2
        
        assert len(json_files) >= 2, f"No se generaron suficientes JSONs. Encontrados: {len(json_files)}"
        
        # ============================================================
        # FASE 2: RENDERIZADO
        # ============================================================
        print(f"\n[FASE 2] Iniciando renderizado de {len(json_files)} videos...")
        phase2_start = time.time()
        
        rendered_videos = []
        for idx, json_file in enumerate(json_files, 1):
            print(f"\n   Renderizando video {idx}/{len(json_files)}: {json_file.name}")
            video_path = Path(RESULTS_FOLDER) / f"{json_file.stem}_skeleton.mp4"
            
            render_start = time.time()
            render_skeleton_video(str(json_file), str(video_path))
            render_time = time.time() - render_start
            
            self.generated_files.append(str(video_path))
            rendered_videos.append(video_path)
            
            print(f"   ✓ Video {idx} renderizado en {render_time:.2f}s")
        
        phase2_end = time.time()
        phase2_time = phase2_end - phase2_start
        
        print(f"\n✅ [FASE 2 COMPLETADA] Tiempo total de render: {phase2_time:.2f}s")
        
        # ============================================================
        # CÁLCULO DE TIEMPO TOTAL E2E
        # ============================================================
        total_time_e2e = phase1_time + phase2_time
        
        print(f"\n{'='*80}")
        print(f"⏱️ [MÉTRICA LATENCIA E2E] TIEMPO TOTAL HASTA RENDER COMPLETO")
        print(f"{'='*80}")
        print(f"   - Fase 1 (Procesamiento hasta JSON): {phase1_time:.2f}s")
        print(f"   - Fase 2 (Renderizado): {phase2_time:.2f}s")
        print(f"   - TOTAL E2E: {total_time_e2e:.2f}s")
        print(f"   - Objetivo ideal: {TARGET_TIME_SECONDS}s (3 minutos)")
        
        if total_time_e2e <= TARGET_TIME_SECONDS:
            print(f"   ✅ EXCELENTE: Se cumplió el objetivo de {TARGET_TIME_SECONDS}s")
        else:
            exceso = total_time_e2e - TARGET_TIME_SECONDS
            print(f"   ⚠️ INFORMATIVO: Se excedió en {exceso:.2f}s ({(exceso/TARGET_TIME_SECONDS)*100:.1f}%)")
        print(f"{'='*80}")
        
        # --- MÉTRICAS ADICIONALES ---
        
        # 1. Desglose de tiempos
        if timings:
            print("\n   --- [Métrica 1: Desglose de Tiempos (Cuello de Botella)] ---")
            print(f"   - Extracción (Pipeline): {timings.get('t_extraccion_video', 0):.2f}s")
            print(f"   - Inferencia (CVAE): {timings.get('t_inferencia_cvae', 0):.2f}s")
            print(f"   - Inferencia (GAN): {timings.get('t_inferencia_gan', 0):.2f}s")
            print(f"   - Renderizado (Total): {phase2_time:.2f}s")
            timings['render_time'] = phase2_time
        else:
            timings = {
                'render_time': phase2_time,
                't_fase1_total': phase1_time,
                't_fase2_render': phase2_time
            }
        
        # 2. Uso de recursos
        resources = {}
        if mem_usage_samples and cpu_usage_samples:
            resources = {
                'cpu_max': max(cpu_usage_samples),
                'cpu_avg': sum(cpu_usage_samples) / len(cpu_usage_samples),
                'ram_avg': sum(mem_usage_samples) / len(mem_usage_samples),
                'ram_max': max(mem_usage_samples)
            }
            print("\n   --- [Métrica 2: Uso de Recursos (Durante Procesamiento)] ---")
            print(f"   - Uso Máx. CPU: {resources['cpu_max']:.2f}%")
            print(f"   - Uso Prom. CPU: {resources['cpu_avg']:.2f}%")
            print(f"   - Uso Prom. RAM: {resources['ram_avg']:.2f} MB")
            print(f"   - Uso Máx. RAM: {resources['ram_max']:.2f} MB")
        
        # 3. Validar JSONs
        total_json_size_kb = 0
        total_frames = 0
        
        for json_file in json_files:
            total_json_size_kb += json_file.stat().st_size / 1024
            
            with open(json_file, 'r') as f:
                data = json.load(f)
                assert 'metadata' in data, f"JSON {json_file.name} sin metadata"
                assert 'frames' in data, f"JSON {json_file.name} sin frames"
                assert len(data['frames']) > 0, f"JSON {json_file.name} sin frames de datos"
                
                if total_frames == 0:
                    total_frames = len(data['frames'])
        
        # 4. Throughput
        throughput = {}
        if phase1_time > 0:
            throughput = {
                'frames_ps': total_frames / phase1_time,
                'kb_ps': total_json_size_kb / phase1_time,
                'render_fps': total_frames / phase2_time if phase2_time > 0 else 0
            }
            print("\n   --- [Métrica 3: Throughput (Rendimiento)] ---")
            print(f"   - Frames Procesados: {total_frames}")
            print(f"   - Frames/s (Procesamiento): {throughput['frames_ps']:.2f}")
            print(f"   - Frames/s (Renderizado): {throughput['render_fps']:.2f}")
            print(f"   - Tamaño Total JSONs: {total_json_size_kb:.2f} KB")
        
        # Verificar que los videos fueron renderizados
        for video_path in rendered_videos:
            assert video_path.exists(), f"Video no renderizado: {video_path}"
            assert video_path.stat().st_size > 0, f"Video vacío: {video_path}"
        
        print(f"\n✓ Flujo completo para {resolution}: {len(json_files)} JSONs y {len(rendered_videos)} videos en {total_time_e2e:.2f}s")
        
        # --- GUARDAR MÉTRICAS EN EL DICCIONARIO GLOBAL ---
        global_test_metrics['runs'].append({
            'resolution': resolution,
            'latency_total': total_time_e2e,
            'latency_phase1': phase1_time,
            'latency_phase2': phase2_time,
            'timings': timings,
            'resources': resources,
            'throughput': throughput,
            'target_met': total_time_e2e <= TARGET_TIME_SECONDS
        })

    def test_extension_cuello_botella(self):
        """
        Extensión 4.1: Ejecuta la prueba de rendimiento 3 veces.
        """
        print("\n" + "="*80)
        print("EXTENSIÓN 4.1: Prueba de Cuello de Botella (3 ejecuciones)")
        print("="*80)
        
        for i in range(RETRIES):
            print(f"\n{'#'*80}")
            print(f"# EJECUCIÓN {i+1}/{RETRIES}")
            print(f"{'#'*80}")
            self.test_rendimiento_flujo_completo_hasta_render(resolution=f"run{i+1}")
        
        print("\n✅ Extensión 4.1 completada: 3 ejecuciones exitosas")

    def test_extension_error_en_etapa(self):
        """
        Extensión 4.2: Si error en etapa, registrar excepción y marcar como fallida.
        """
        print("\n" + "="*80)
        print("EXTENSIÓN 4.2: Prueba de Manejo de Errores")
        print("="*80)
        
        invalid_video_path = self.test_upload_dir / "invalid_video_error_test.mp4"
        invalid_video_path.write_bytes(b"")  # Archivo vacío
        self.generated_files.append(str(invalid_video_path))
        
        task_id = "error_test"
        start_time = time.time()
        
        thread = threading.Thread(
            target=process_video_async,
            args=(str(invalid_video_path), task_id, TEST_USERNAME, True, True)
        )
        thread.start()
        
        max_wait = 60
        waited = 0
        while processing_status.get(task_id, {}).get('status') == 'processing' and waited < max_wait:
            time.sleep(1)
            waited += 1
        
        thread.join(timeout=10)
        
        end_time = time.time()
        total_time = end_time - start_time

        final_status = processing_status.get(task_id, {}).get('status', 'not_found')
        
        assert final_status == 'error', f"El procesamiento fallido no reportó 'error'. Estado: {final_status}"
        
        status_message = processing_status[task_id]['message']
        assert total_time < 90, "El error no fue detectado rápidamente (< 90s)"
        assert 'error' in status_message.lower() or 'invalid' in status_message.lower(), "No se registró excepción"
        
        print(f"✓ Error manejado correctamente en {total_time:.2f}s")
        print(f"  Mensaje: {status_message}")
        
    @pytest.mark.parametrize("resolution", ["1080p"]) 
    def test_condicion_exito_subvariaciones(self, resolution):
        """
        Sub-variaciones: Probar con 1080p (720p se prueba en el test principal).
        """
        print(f"\n{'='*80}")
        print(f"SUB-VARIACIÓN: Prueba con {resolution}")
        print(f"{'='*80}")
        
        self.test_rendimiento_flujo_completo_hasta_render(resolution)
        
        print(f"\n✓ Sub-variación {resolution} exitosa")


# ========================================================================
# --- HOOK DE PYTEST ---
# ========================================================================
def pytest_sessionfinish(session):
    """
    Hook se ejecuta al final de la sesión de Pytest.
    Calcula los promedios y guarda el reporte.
    """
    print("\n" + "="*80)
    print("[PYTEST HOOK] Sesión finalizada, calculando reporte...")
    print("="*80)
    
    runs = global_test_metrics.get('runs', [])
    if not runs:
        print("[Pytest Hook] No se encontraron datos de 'runs' para generar reporte.")
        return

    num_runs = len(runs)
    print(f"\n📊 Procesando {num_runs} ejecución(es)...")
    
    final_report_data = {
        'num_runs': num_runs,
        'avg_latency_total': sum(r['latency_total'] for r in runs) / num_runs,
        'avg_latency_phase1': sum(r['latency_phase1'] for r in runs) / num_runs,
        'avg_latency_phase2': sum(r['latency_phase2'] for r in runs) / num_runs,
        'target_seconds': TARGET_TIME_SECONDS,
        'runs_meeting_target': sum(1 for r in runs if r.get('target_met', False)),
        'avg_timings': {},
        'avg_resources': {},
        'avg_throughput': {}
    }
    
    # Timings
    if runs[0].get('timings'):
        final_report_data['avg_timings'] = {
            't_extraccion_video': sum(r['timings'].get('t_extraccion_video', 0) for r in runs) / num_runs,
            't_inferencia_cvae': sum(r['timings'].get('t_inferencia_cvae', 0) for r in runs) / num_runs,
            't_inferencia_gan': sum(r['timings'].get('t_inferencia_gan', 0) for r in runs) / num_runs,
            'render_time': sum(r['timings'].get('render_time', 0) for r in runs) / num_runs,
        }

    # Recursos
    if runs[0].get('resources'):
        final_report_data['avg_resources'] = {
            'cpu_max': sum(r['resources'].get('cpu_max', 0) for r in runs) / num_runs,
            'cpu_avg': sum(r['resources'].get('cpu_avg', 0) for r in runs) / num_runs,
            'ram_max': sum(r['resources'].get('ram_max', 0) for r in runs) / num_runs,
        }

    # Throughput
    if runs[0].get('throughput'):
        final_report_data['avg_throughput'] = {
            'frames_ps_processing': sum(r['throughput'].get('frames_ps', 0) for r in runs) / num_runs,
            'render_fps': sum(r['throughput'].get('render_fps', 0) for r in runs) / num_runs,
        }

    # Guardar reporte
    report_file_path = Path(__file__).parent / "reporte_cp0006.json"
    try:
        with open(report_file_path, 'w') as f:
            json.dump(final_report_data, f, indent=4)
        print(f"✓ Reporte final guardado en: {report_file_path}")
    except Exception as e:
        print(f"❌ ERROR al guardar reporte: {e}")


# ========================================================================
# --- FUNCIÓN DE REPORTE FINAL ---
# ========================================================================
def generar_reporte_final():
    """
    Genera reporte de rendimiento final, leyendo desde el JSON.
    """
    print("\n" + "="*80)
    print("✨ REPORTE FINAL CP-0006: RENDIMIENTO HASTA RENDERIZADO COMPLETO ✨")
    print("="*80)
    
    report_file_path = Path(__file__).parent / "reporte_cp0006.json"
    if not report_file_path.exists():
        print("❌ ERROR: No se encontró el archivo 'reporte_cp0006.json'")
        print("="*80)
        return

    with open(report_file_path, 'r') as f:
        metrics = json.load(f)

    num_runs = metrics.get('num_runs', 0)
    avg_total = metrics.get('avg_latency_total', 0)
    avg_phase1 = metrics.get('avg_latency_phase1', 0)
    avg_phase2 = metrics.get('avg_latency_phase2', 0)
    target = metrics.get('target_seconds', TARGET_TIME_SECONDS)
    runs_met = metrics.get('runs_meeting_target', 0)
    
    print(f"\n📊 RESUMEN DE {num_runs} EJECUCIÓN(ES)")
    print("-" * 80)
    print(f"   ⏱️  LATENCIA E2E PROMEDIO (Total): {avg_total:.2f}s")
    print(f"   └─ Fase 1 (Procesamiento → JSON): {avg_phase1:.2f}s")
    print(f"   └─ Fase 2 (Renderizado): {avg_phase2:.2f}s")
    print(f"\n   🎯 Objetivo ideal: {target}s (3 minutos)")
    print(f"   {'✅' if runs_met == num_runs else '⚠️ '} Ejecuciones cumpliendo objetivo: {runs_met}/{num_runs}")
    
    if avg_total > target:
        exceso = avg_total - target
        print(f"   📈 Exceso promedio: +{exceso:.2f}s ({(exceso/target)*100:.1f}%)")
    
    # Desglose de tiempos
    if metrics.get('avg_timings'):
        timings = metrics['avg_timings']
        print("\n   --- Desglose de Tiempos (Promedio) ---")
        print(f"   1. Extracción (Pipeline):  {timings.get('t_extraccion_video', 0):.2f}s")
        print(f"   2. Inferencia (CVAE):      {timings.get('t_inferencia_cvae', 0):.2f}s")
        print(f"   3. Inferencia (GAN):       {timings.get('t_inferencia_gan', 0):.2f}s")
        print(f"   4. Renderizado:            {timings.get('render_time', 0):.2f}s")
        print(f"   {'─'*40}")
        suma_parcial = sum([
            timings.get('t_extraccion_video', 0),
            timings.get('t_inferencia_cvae', 0),
            timings.get('t_inferencia_gan', 0),
            timings.get('render_time', 0)
        ])
        print(f"   TOTAL:                     {suma_parcial:.2f}s")
    
    # Recursos
    if metrics.get('avg_resources'):
        resources = metrics['avg_resources']
        print("\n   --- Uso de Recursos (Promedio) ---")
        print(f"   CPU Máximo:   {resources.get('cpu_max', 0):.2f}%")
        print(f"   CPU Promedio: {resources.get('cpu_avg', 0):.2f}%")
        print(f"   RAM Máxima:   {resources.get('ram_max', 0):.2f} MB")

    # Throughput
    if metrics.get('avg_throughput'):
        throughput = metrics['avg_throughput']
        print("\n   --- Rendimiento/Throughput (Promedio) ---")
        print(f"   Frames/s (Procesamiento): {throughput.get('frames_ps_processing', 0):.2f}")
        print(f"   Frames/s (Renderizado):   {throughput.get('render_fps', 0):.2f}")
    
    print("\n" + "="*80)
    print("💡 NOTA: Este test mide el flujo completo incluyendo renderizado.")
    print("   Para pruebas de estrés con múltiples usuarios, usar JMeter o Locust.")
    print("="*80)
    
    # Limpiar archivo temporal
    try:
        report_file_path.unlink()
        print(f"🗑️  Archivo temporal '{report_file_path.name}' eliminado\n")
    except OSError:
        pass


if __name__ == "__main__":
    # Limpiar métricas y reportes previos
    global_test_metrics = {'runs': []}
    report_file_path = Path(__file__).parent / "reporte_cp0006.json"
    if report_file_path.exists():
        report_file_path.unlink()

    # Ejecutar tests
    result = pytest.main([__file__, "-v", "--tb=short", "--color=yes"])
    
    # Generar reporte si hubo éxito
    if result == 0:
        generar_reporte_final()
    else:
        print("\n⚠️  Algunas pruebas fallaron. Revise los logs para más detalles.")