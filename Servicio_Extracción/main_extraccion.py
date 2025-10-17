import os
import glob
import time
from datetime import datetime
from Pipeline import Pipeline

def process_all_videos():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    bd_folder = os.path.join(project_root, "BD")
    
    if not os.path.exists(bd_folder):
        raise FileNotFoundError(f"No se encontró la carpeta BD en: {bd_folder}")
    
    video_extensions = ['*.mp4', '*.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(bd_folder, ext)))
    
    if not video_files:
        raise FileNotFoundError(f"No se encontraron archivos de video en: {bd_folder}")
    
    print(f"=== PROCESAMIENTO MASIVO DE VIDEOS ===")
    print(f"Fecha de inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total de videos encontrados: {len(video_files)}")
    print(f"Configuración: Suavizado=ON, Corrección de piernas=ON")
    print("=" * 50)
    
    # Estadísticas generales
    successful_extractions = 0
    failed_extractions = 0
    total_processing_time = 0
    failed_videos = []
    
    for i, video_file in enumerate(video_files, 1):
        video_name = os.path.basename(video_file)
        print(f"\n[{i}/{len(video_files)}] Procesando: {video_name}")
        
        try:
            start_time = time.time()
            
            # Crear pipeline para este video
            pipeline = Pipeline(video_file, smooth_enabled=True, fix_legs=True)
            df_raw, df_processed = pipeline.run()
            
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            print(f"[OK] Exito - Tiempo: {processing_time:.2f}s - Frames: {len(df_processed)}")
            successful_extractions += 1
            
        except Exception as e:
            print(f"[ERROR] Error procesando {video_name}: {str(e)}")
            failed_extractions += 1
            failed_videos.append(video_name)
            continue
    
    # Estadísticas finales
    print("\n" + "=" * 50)
    print("=== ESTADÍSTICAS FINALES ===")
    print(f"Videos procesados exitosamente: {successful_extractions}")
    print(f"Videos con errores: {failed_extractions}")
    print(f"Tiempo total de procesamiento: {total_processing_time:.2f}s")
    print(f"Tiempo promedio por video: {total_processing_time/len(video_files):.2f}s")
    
    if failed_videos:
        print(f"\nVideos que fallaron:")
        for video in failed_videos:
            print(f"  - {video}")
    
    print(f"\nFecha de finalización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    return successful_extractions, failed_extractions

if __name__ == "__main__":
    try:
        successful, failed = process_all_videos()
        
        if failed == 0:
            print("\n[EXITO] ¡Todos los videos se procesaron exitosamente!")
        elif successful > 0:
            print(f"\n[PARCIAL] Se procesaron {successful} videos, {failed} fallaron.")
        else:
            print("\n[FALLO] No se pudo procesar ningún video.")
            
    except Exception as e:
        print(f"\n[CRITICO] Error critico: {str(e)}")
        raise
