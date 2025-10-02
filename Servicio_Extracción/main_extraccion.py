import os
from Pipeline import Pipeline

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    video_file = os.path.join(project_root, "B_D", "Paso 2 - Hombre - Carnaval.mp4")

    if not os.path.exists(video_file):
        raise FileNotFoundError(f"No se encontró el video en: {video_file}")

    pipeline = Pipeline(video_file, smooth_enabled=True, fix_legs=True)
    df_raw, df_processed = pipeline.run()

    print("\nVista previa del CSV procesado:")
    print(df_processed.head(10))
