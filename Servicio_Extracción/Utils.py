def verify_extraction(df, total_frames):
    """Verifica si la extracción fue exitosa."""
    if len(df) == 0:
        print("ERROR: No se detectó ninguna pose en el video")
        return False
    detection_rate = (len(df) / total_frames) * 100
    print(f"Tasa de detección: {detection_rate:.1f}%")
    return True


def print_final_statistics(df_raw, df_processed, fps, total_frames):
    """Imprime estadísticas comparativas antes y después del procesado."""
    print("ESTADÍSTICAS FINALES")
    print(f"Frames procesados: {len(df_processed)}/{total_frames}")
    raw_vis = df_raw.filter(like="_visibility").mean().mean()
    proc_vis = df_processed.filter(like="_visibility").mean().mean()
    print(f"Visibilidad antes: {raw_vis:.3f}, después: {proc_vis:.3f}")
