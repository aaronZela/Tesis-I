import pyvista as pv
import numpy as np
import json
import os
import cv2
import tempfile

# Conexiones del esqueleto de MediaPipe Pose
POSE_CONNECTIONS = [
    # Cara
    ('NOSE', 'LEFT_EYE_INNER'), ('LEFT_EYE_INNER', 'LEFT_EYE'),
    ('LEFT_EYE', 'LEFT_EYE_OUTER'), ('LEFT_EYE_OUTER', 'LEFT_EAR'),
    ('NOSE', 'RIGHT_EYE_INNER'), ('RIGHT_EYE_INNER', 'RIGHT_EYE'),
    ('RIGHT_EYE', 'RIGHT_EYE_OUTER'), ('RIGHT_EYE_OUTER', 'RIGHT_EAR'),
    ('MOUTH_LEFT', 'MOUTH_RIGHT'),
    # Torso
    ('LEFT_SHOULDER', 'RIGHT_SHOULDER'),
    ('LEFT_SHOULDER', 'LEFT_HIP'), ('RIGHT_SHOULDER', 'RIGHT_HIP'),
    ('LEFT_HIP', 'RIGHT_HIP'),
    # Brazos
    ('LEFT_SHOULDER', 'LEFT_ELBOW'), ('LEFT_ELBOW', 'LEFT_WRIST'),
    ('LEFT_WRIST', 'LEFT_PINKY'), ('LEFT_WRIST', 'LEFT_INDEX'),
    ('LEFT_WRIST', 'LEFT_THUMB'),
    ('RIGHT_SHOULDER', 'RIGHT_ELBOW'), ('RIGHT_ELBOW', 'RIGHT_WRIST'),
    ('RIGHT_WRIST', 'RIGHT_PINKY'), ('RIGHT_WRIST', 'RIGHT_INDEX'),
    ('RIGHT_WRIST', 'RIGHT_THUMB'),
    # Piernas
    ('LEFT_HIP', 'LEFT_KNEE'), ('LEFT_KNEE', 'LEFT_ANKLE'),
    ('LEFT_ANKLE', 'LEFT_HEEL'), ('LEFT_ANKLE', 'LEFT_FOOT_INDEX'),
    ('RIGHT_HIP', 'RIGHT_KNEE'), ('RIGHT_KNEE', 'RIGHT_ANKLE'),
    ('RIGHT_ANKLE', 'RIGHT_HEEL'), ('RIGHT_ANKLE', 'RIGHT_FOOT_INDEX'),
]

def create_scene():
    """Crear escena 3D con PyVista"""
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    plotter.set_background('white')
    
    # Iluminación mejorada
    light1 = pv.Light(position=(2, 2, 2), light_type='camera light')
    light2 = pv.Light(position=(-2, 2, 2), light_type='camera light', intensity=0.5)
    plotter.add_light(light1)
    plotter.add_light(light2)
    
    return plotter

def normalize_coordinates(keypoints_dict):
    """Normalizar coordenadas para centrar en el origen"""
    # Calcular centro usando caderas
    if 'LEFT_HIP' in keypoints_dict and 'RIGHT_HIP' in keypoints_dict:
        center_x = (keypoints_dict['LEFT_HIP']['x'] + keypoints_dict['RIGHT_HIP']['x']) / 2
        center_y = (keypoints_dict['LEFT_HIP']['y'] + keypoints_dict['RIGHT_HIP']['y']) / 2
        center_z = (keypoints_dict['LEFT_HIP']['z'] + keypoints_dict['RIGHT_HIP']['z']) / 2
    else:
        # Fallback: usar promedio de todos los puntos
        center_x = np.mean([kp['x'] for kp in keypoints_dict.values()])
        center_y = np.mean([kp['y'] for kp in keypoints_dict.values()])
        center_z = np.mean([kp['z'] for kp in keypoints_dict.values()])
    
    # Centrar y escalar (MediaPipe usa coordenadas normalizadas 0-1)
    normalized = {}
    scale = 5.0  # Escala para hacer el esqueleto más visible
    
    for name, kp in keypoints_dict.items():
        normalized[name] = {
            'x': (kp['x'] - center_x) * scale,
            'y': -(kp['y'] - center_y) * scale,  # Invertir Y para orientación correcta
            'z': (kp['z'] - center_z) * scale,
            'visibility': kp.get('visibility', 1.0)
        }
    
    return normalized

def render_frame(plotter, frame_data, frame_num, total_frames, temp_dir):
    """Renderizar un frame con el esqueleto"""
    plotter.clear()
    
    # Convertir keypoints a diccionario
    keypoints_dict = {kp['name']: kp for kp in frame_data.get('keypoints', [])}
    
    if not keypoints_dict:
        print(f"[WARNING] Frame {frame_num}: No hay keypoints")
        return None
    
    # Normalizar coordenadas
    normalized_kp = normalize_coordinates(keypoints_dict)
    
    # Renderizar keypoints como esferas
    for name, kp in normalized_kp.items():
        if kp['visibility'] > 0.5:  # Solo renderizar puntos visibles
            sphere = pv.Sphere(radius=0.08, center=(kp['x'], kp['y'], kp['z']))
            
            # Color según parte del cuerpo
            if 'EYE' in name or 'EAR' in name or 'NOSE' in name or 'MOUTH' in name:
                color = [0.2, 0.6, 1.0]  # Azul para cara
            elif 'SHOULDER' in name or 'HIP' in name:
                color = [1.0, 0.2, 0.2]  # Rojo para torso
            elif 'ELBOW' in name or 'WRIST' in name or 'THUMB' in name or 'INDEX' in name or 'PINKY' in name:
                color = [0.2, 1.0, 0.2]  # Verde para brazos/manos
            else:
                color = [1.0, 0.8, 0.2]  # Amarillo para piernas/pies
            
            plotter.add_mesh(sphere, color=color, smooth_shading=True)
    
    # Renderizar conexiones como líneas
    for start_name, end_name in POSE_CONNECTIONS:
        if start_name in normalized_kp and end_name in normalized_kp:
            start_kp = normalized_kp[start_name]
            end_kp = normalized_kp[end_name]
            
            # Solo dibujar si ambos puntos son visibles
            if start_kp['visibility'] > 0.5 and end_kp['visibility'] > 0.5:
                points = np.array([
                    [start_kp['x'], start_kp['y'], start_kp['z']],
                    [end_kp['x'], end_kp['y'], end_kp['z']]
                ])
                line = pv.Line(points[0], points[1])
                plotter.add_mesh(line, color=[0.3, 0.3, 0.3], line_width=5)
    
    # Configurar cámara
    plotter.camera_position = [
        (0, 0, 10),   # Posición de la cámara
        (0, 0, 0),     # Punto focal (centro)
        (0, 1, 0)      # Vector up
    ]
    
    # Capturar frame
    frame_path = os.path.join(temp_dir, f'frame_{frame_num:04d}.png')
    plotter.screenshot(frame_path)
    
    return frame_path

def create_video_from_frames(frame_paths, output_path, fps=30):
    """Crear video MP4 desde frames usando codec mp4v (más compatible en Windows)"""
    print(f"[VIDEO] Creando video desde {len(frame_paths)} frames...")
    
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print(f"[ERROR] No se pudo leer el primer frame: {frame_paths[0]}")
        return
    
    height, width, layers = first_frame.shape
    
    # Usar codec mp4v para mayor compatibilidad con reproductores de Windows
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("[ERROR] No se pudo crear el video writer con codec mp4v")
        return
    
    for i, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[WARNING] No se pudo leer frame {i}: {frame_path}")
            continue
        
        # Asegurar que el frame tenga el tamaño correcto
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))
        
        video_writer.write(frame)
        
        if (i + 1) % 10 == 0:
            progress = int((i + 1) / len(frame_paths) * 100)
            print(f"[VIDEO] Codificando: {progress}% ({i + 1}/{len(frame_paths)})")
    
    video_writer.release()
    print(f"[VIDEO] ✓ Video guardado: {output_path}")

def calculate_global_center(frames):
    """Calcular el centro global de todos los frames para mantener consistencia"""
    all_centers_x = []
    all_centers_y = []
    all_centers_z = []
    
    for frame_data in frames:
        keypoints_dict = {kp['name']: kp for kp in frame_data.get('keypoints', [])}
        if 'LEFT_HIP' in keypoints_dict and 'RIGHT_HIP' in keypoints_dict:
            center_x = (keypoints_dict['LEFT_HIP']['x'] + keypoints_dict['RIGHT_HIP']['x']) / 2
            center_y = (keypoints_dict['LEFT_HIP']['y'] + keypoints_dict['RIGHT_HIP']['y']) / 2
            center_z = (keypoints_dict['LEFT_HIP']['z'] + keypoints_dict['RIGHT_HIP']['z']) / 2
            all_centers_x.append(center_x)
            all_centers_y.append(center_y)
            all_centers_z.append(center_z)
    
    if all_centers_x:
        return (
            np.mean(all_centers_x),
            np.mean(all_centers_y),
            np.mean(all_centers_z)
        )
    return (0.5, 0.5, 0.0)  # Default center

def normalize_coordinates_global(keypoints_dict, global_center, scale=5.0):
    """Normalizar coordenadas usando un centro global para mantener movimiento relativo"""
    normalized = {}
    
    for name, kp in keypoints_dict.items():
        normalized[name] = {
            'x': (kp['x'] - global_center[0]) * scale,
            'y': -(kp['y'] - global_center[1]) * scale,  # Invertir Y para orientación correcta
            'z': (kp['z'] - global_center[2]) * scale,
            'visibility': kp.get('visibility', 1.0)
        }
    
    return normalized

def render_skeleton_video(json_path, output_path, max_frames=None):
    """Función principal para renderizar video de esqueleto"""
    print(f"[RENDER] Cargando: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    frames = data.get('frames', [])
    metadata = data.get('metadata', {})
    fps = metadata.get('fps', 30)
    
    if not frames:
        print("[ERROR] No se encontraron frames en el JSON")
        return
    
    # Limitar frames si se especifica
    if max_frames:
        frames = frames[:max_frames]
    
    total_frames = len(frames)
    print(f"[RENDER] Configuración: {total_frames} frames @ {fps} FPS")
    
    # Calcular centro global para mantener movimiento relativo
    global_center = calculate_global_center(frames)
    print(f"[RENDER] Centro global calculado: {global_center}")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"[RENDER] Renderizando frames...")
        
        plotter = create_scene()
        frame_paths = []
        
        for i, frame_data in enumerate(frames):
            # Convertir keypoints a diccionario
            keypoints_dict = {kp['name']: kp for kp in frame_data.get('keypoints', [])}
            
            if not keypoints_dict:
                print(f"[WARNING] Frame {i}: No hay keypoints")
                continue
            
            # Normalizar usando centro global
            normalized_kp = normalize_coordinates_global(keypoints_dict, global_center)
            
            # Renderizar frame
            plotter.clear()
            
            # Renderizar keypoints como esferas
            for name, kp in normalized_kp.items():
                if kp['visibility'] > 0.5:
                    sphere = pv.Sphere(radius=0.08, center=(kp['x'], kp['y'], kp['z']))
                    
                    # Color según parte del cuerpo
                    if 'EYE' in name or 'EAR' in name or 'NOSE' in name or 'MOUTH' in name:
                        color = [0.2, 0.6, 1.0]  # Azul para cara
                    elif 'SHOULDER' in name or 'HIP' in name:
                        color = [1.0, 0.2, 0.2]  # Rojo para torso
                    elif 'ELBOW' in name or 'WRIST' in name or 'THUMB' in name or 'INDEX' in name or 'PINKY' in name:
                        color = [0.2, 1.0, 0.2]  # Verde para brazos/manos
                    else:
                        color = [1.0, 0.8, 0.2]  # Amarillo para piernas/pies
                    
                    plotter.add_mesh(sphere, color=color, smooth_shading=True)
            
            # Renderizar conexiones como líneas
            for start_name, end_name in POSE_CONNECTIONS:
                if start_name in normalized_kp and end_name in normalized_kp:
                    start_kp = normalized_kp[start_name]
                    end_kp = normalized_kp[end_name]
                    
                    if start_kp['visibility'] > 0.5 and end_kp['visibility'] > 0.5:
                        points = np.array([
                            [start_kp['x'], start_kp['y'], start_kp['z']],
                            [end_kp['x'], end_kp['y'], end_kp['z']]
                        ])
                        line = pv.Line(points[0], points[1])
                        plotter.add_mesh(line, color=[0.3, 0.3, 0.3], line_width=5)
            
            # Configurar cámara
            plotter.camera_position = [
                (0, 0, 10),   # Posición de la cámara
                (0, 0, 0),     # Punto focal (centro)
                (0, 1, 0)      # Vector up
            ]
            
            # Capturar frame
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.png')
            plotter.screenshot(frame_path)
            frame_paths.append(frame_path)
            
            if (i + 1) % 30 == 0 or i == total_frames - 1:
                progress = int((i + 1) / total_frames * 100)
                print(f"[RENDER] Progreso: {progress}% ({i + 1}/{total_frames})")
        
        plotter.close()
        
        if not frame_paths:
            print("[ERROR] No se generaron frames válidos")
            return
        
        create_video_from_frames(frame_paths, output_path, fps)
    
    print(f"[RENDER] ✓ Completado: {output_path}")

def main():
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python render_skeleton_video.py input.json [output.mp4] [max_frames]")
        print("\nEjemplo: python render_skeleton_video.py pose_data.json output.mp4 120")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"Error: Archivo no encontrado: {json_path}")
        sys.exit(1)
    
    # Output path
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        videos_dir = os.path.join(script_dir, '..', 'Rendered_Videos')
        os.makedirs(videos_dir, exist_ok=True)
        
        json_filename = os.path.basename(json_path)
        video_filename = json_filename.replace('.json', '_skeleton.mp4')
        output_path = os.path.join(videos_dir, video_filename)
    
    # Max frames (opcional)
    max_frames = None
    if len(sys.argv) >= 4:
        max_frames = int(sys.argv[3])
    
    print(f"[INFO] Video se guardará en: {output_path}")
    
    render_skeleton_video(json_path, output_path, max_frames)

if __name__ == "__main__":
    main()