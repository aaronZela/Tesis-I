import os
from Procesar_video import VideoProcessor
from Procesar import fix_low_visibility, smooth, clip_coordinates, add_quality
from Utils import verify_extraction, print_final_statistics


class Pipeline:
    def __init__(self, video_path, smooth_enabled=True, fix_legs=True):
        self.video_path = video_path
        self.smooth_enabled = smooth_enabled
        self.fix_legs = fix_legs

    def run(self):
        # Preparar rutas de salida basadas en el nombre del video
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "Coordenadas_csv")
        os.makedirs(output_dir, exist_ok=True)

        video_basename = os.path.splitext(os.path.basename(self.video_path))[0]
        raw_path = os.path.join(output_dir, f"{video_basename}_raw.csv")
        processed_path = os.path.join(output_dir, f"{video_basename}_processed.csv")

        processor = VideoProcessor(self.video_path)
        df_raw, fps, total_frames = processor.extract_coordinates(output_csv=raw_path)

        verify_extraction(df_raw, total_frames)

        df_processed = df_raw.copy()
        if self.fix_legs:
            df_processed = fix_low_visibility(df_processed)
        if self.smooth_enabled:
            df_processed = smooth(df_processed)

        df_processed = clip_coordinates(df_processed)
        df_processed = add_quality(df_processed)

        df_processed.to_csv(processed_path, index=False)
        print_final_statistics(df_raw, df_processed, fps, total_frames)

        return df_raw, df_processed
