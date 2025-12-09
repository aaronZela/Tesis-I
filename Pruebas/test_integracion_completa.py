import os
import sys
import pytest
import pandas as pd
import json
from datetime import datetime
import tempfile

# Agregar paths para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend', 'Servicio_extraccion'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Backend', 'Servicio_IA'))

# Imports necesarios
from Servicio_extraccion.Pipeline import Pipeline
from Servicio_IA.generate_cvae_lstm2 import generate_cvae_sample
from Servicio_IA.generate_gan_lstm2 import generate_gan_sample

# Path a un video de prueba (ajusta si es necesario)
SAMPLE_VIDEO_PATH = os.path.join(os.path.dirname(__file__), '..', 'BD', 'Paso 1 - Hombre - Carnaval.mp4')
MODEL_CVAE_PATH = os.path.join(os.path.dirname(__file__), '..', 'Backend', 'Servicio_IA_Entrenada', 'cvae_lstm_best.pt')
MODEL_GAN_PATH = os.path.join(os.path.dirname(__file__), '..', 'Backend', 'Servicio_IA_Entrenada', 'gan_lstm_best_model.pt')

def test_integracion_extraccion_ia_json():
    """
    Prueba de integración completa:
    1. Extracción de keypoints de un video de muestra
    2. Generación de secuencias con CVAE y GAN
    3. Verificación de generación de JSONs
    """
    if not os.path.exists(SAMPLE_VIDEO_PATH):
        pytest.skip(f"Video de muestra no encontrado: {SAMPLE_VIDEO_PATH}")
    
    print(f"Usando video de prueba: {SAMPLE_VIDEO_PATH}")
    
    # 1. Extracción de keypoints usando Pipeline
    print("Paso 1: Iniciando extracción de keypoints...")
    pipeline = Pipeline(
        video_path=SAMPLE_VIDEO_PATH,
        smooth_enabled=True,
        fix_legs=True
    )
    df_raw, df_processed = pipeline.run()
    
    assert df_raw is not None, "Fallo en extracción raw"
    assert df_processed is not None, "Fallo en extracción processed"
    assert len(df_processed) > 0, "DataFrame processed vacío"
    
    print(f"✓ Extracción completada: {len(df_processed)} frames procesados")
    
    # Guardar temporalmente los CSVs para simular el flujo
    with tempfile.TemporaryDirectory() as temp_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = "test_video"
        raw_filename = f"test_raw_{timestamp}.csv"
        processed_filename = f"test_processed_{timestamp}.csv"
        
        raw_path = os.path.join(temp_dir, raw_filename)
        processed_path = os.path.join(temp_dir, processed_filename)
        
        df_raw.to_csv(raw_path, index=False)
        df_processed.to_csv(processed_path, index=False)
        
        print(f"CSVs temporales guardados en: {temp_dir}")
        
        # 2. Extraer etiquetas del CSV procesado (simulando detección)
        # Usar modos (valores más frecuentes) o valores por defecto
        paso = int(df_processed['paso'].mode().iloc[0]) if 'paso' in df_processed.columns and not df_processed['paso'].mode().empty else 1
        genero = str(df_processed['genero'].mode().iloc[0]) if 'genero' in df_processed.columns and not df_processed['genero'].mode().empty else 'Hombre'
        danza = str(df_processed['danza'].mode().iloc[0]) if 'danza' in df_processed.columns and not df_processed['danza'].mode().empty else 'Carnaval'
        
        seq_length = len(df_processed)
        fps = 30  # Valor por defecto
        
        print(f"Etiquetas detectadas: Paso={paso}, Género={genero}, Danza={danza}, Frames={seq_length}")
        
        # 3. Generación con CVAE
        print("Paso 2: Generando secuencia con CVAE...")
        try:
            df_cvae, json_cvae = generate_cvae_sample(
                model_path=MODEL_CVAE_PATH,
                seq_length=seq_length,
                paso=paso,
                genero=genero,
                danza=danza,
                fps=fps
            )
            assert df_cvae is not None, "DataFrame CVAE vacío"
            assert json_cvae is not None, "JSON CVAE vacío"
            
            # Verificar que se generó el JSON
            cvae_json_path = f'generated_cvae_paso{paso}_{genero}_{danza}.json'
            assert os.path.exists(cvae_json_path), f"JSON CVAE no generado: {cvae_json_path}"
            
            # Limpiar archivo generado
            os.remove(cvae_json_path)
            
            print("✓ CVAE generado exitosamente")
        except Exception as e:
            pytest.fail(f"Fallo en generación CVAE: {str(e)}")
        
        # 4. Generación con GAN
        print("Paso 3: Generando secuencia con GAN...")
        try:
            df_gan, json_gan = generate_gan_sample(
                model_path=MODEL_GAN_PATH,
                seq_length=seq_length,
                paso=paso,
                genero=genero,
                danza=danza,
                fps=fps
            )
            assert df_gan is not None, "DataFrame GAN vacío"
            assert json_gan is not None, "JSON GAN vacío"
            
            # Verificar que se generó el JSON
            gan_json_path = f'generated_gan_paso{paso}_{genero}_{danza}.json'
            assert os.path.exists(gan_json_path), f"JSON GAN no generado: {gan_json_path}"
            
            # Limpiar archivo generado
            os.remove(gan_json_path)
            
            print("✓ GAN generado exitosamente")
        except Exception as e:
            pytest.fail(f"Fallo en generación GAN: {str(e)}")
        
        # 5. Verificación final
        print("Paso 4: Verificación de integración completada")
        assert True, "Integración exitosa: Extracción + CVAE + GAN + JSON"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
