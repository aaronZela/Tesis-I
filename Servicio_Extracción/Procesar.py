import numpy as np
from scipy.signal import savgol_filter

def fix_low_visibility(df, threshold=0.5):
    """Corrige articulaciones con baja visibilidad interpolando."""
    df_fixed = df.copy()
    leg_points = ['LEFT_ANKLE', 'RIGHT_ANKLE', 'LEFT_KNEE', 'RIGHT_KNEE',
                  'LEFT_HIP', 'RIGHT_HIP', 'LEFT_HEEL', 'RIGHT_HEEL',
                  'LEFT_FOOT_INDEX', 'RIGHT_FOOT_INDEX']

    for point in leg_points:
        vis_col = f"{point}_visibility"
        if vis_col in df_fixed.columns:
            low_vis_mask = df_fixed[vis_col] < threshold
            for coord in ["x", "y", "z"]:
                coord_col = f"{point}_{coord}"
                df_fixed.loc[low_vis_mask, coord_col] = np.nan
                df_fixed[coord_col] = df_fixed[coord_col].interpolate(
                    method="linear", limit_direction="both"
                )
            df_fixed.loc[low_vis_mask, vis_col] = 0.6
    return df_fixed


def smooth(df, window_length=11, polyorder=3):
    """Suaviza trayectorias con Savitzky-Golay."""
    df_smooth = df.copy()
    if len(df) >= window_length:
        for col in df_smooth.columns:
            if col.endswith(("_x", "_y", "_z")):
                df_smooth[col] = savgol_filter(
                    df[col], window_length=window_length, polyorder=polyorder
                )
    return df_smooth


def clip_coordinates(df):
    """Recorta valores fuera de rango [0,1]."""
    df_clipped = df.copy()
    for coord in ["x", "y", "z"]:
        coord_cols = [c for c in df_clipped.columns if c.endswith(f"_{coord}")]
        df_clipped[coord_cols] = df_clipped[coord_cols].clip(0, 1)
    return df_clipped


def add_quality(df, threshold=0.5):
    """Agrega columna de calidad según visibilidad."""
    df_marked = df.copy()
    for col in df.columns:
        if col.endswith("_visibility"):
            quality_col = col.replace("_visibility", "_quality")
            df_marked[quality_col] = df[col].apply(
                lambda v: "high" if v >= 0.7 else "medium" if v >= threshold else "low"
            )
    return df_marked
