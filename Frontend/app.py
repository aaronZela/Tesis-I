import os
import uuid
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, session
from flask_cors import CORS
from werkzeug.utils import secure_filename
import sys

# Agregar el directorio Backend al path para importar módulos
backend_path = os.path.join(os.path.dirname(__file__), '..', 'Backend')
sys.path.insert(0, backend_path)
sys.path.insert(0, os.path.join(backend_path, 'Servicio_extraccion'))

# Importar módulos directamente
import importlib.util

def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Importar Pipeline
pipeline_path = os.path.join(backend_path, 'Servicio_extraccion', 'Pipeline.py')
Pipeline_module = import_module_from_path('Pipeline', pipeline_path)
Pipeline = Pipeline_module.Pipeline

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'
CORS(app)

# Configuración de archivos
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_FOLDER = os.path.join(os.path.dirname(__file__), 'results')
ALLOWED_EXTENSIONS = {'mp4', 'mov'}

# Crear directorios si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Almacenar el estado de procesamiento
processing_status = {}

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video_async(video_path, task_id, smooth_enabled=True, fix_legs=True):
    """Procesa el video en un hilo separado."""
    try:
        processing_status[task_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Iniciando procesamiento...',
            'start_time': datetime.now().isoformat()
        }
        
        # Crear pipeline y procesar
        pipeline = Pipeline(video_path, smooth_enabled=smooth_enabled, fix_legs=fix_legs)
        df_raw, df_processed = pipeline.run()
        
        # Generar nombres de archivos de salida
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        raw_filename = f"{video_name}_raw_{timestamp}.csv"
        processed_filename = f"{video_name}_processed_{timestamp}.csv"
        
        raw_path = os.path.join(RESULTS_FOLDER, raw_filename)
        processed_path = os.path.join(RESULTS_FOLDER, processed_filename)
        
        # Guardar archivos
        df_raw.to_csv(raw_path, index=False)
        df_processed.to_csv(processed_path, index=False)
        
        processing_status[task_id] = {
            'status': 'completed',
            'progress': 100,
            'message': 'Procesamiento completado exitosamente',
            'end_time': datetime.now().isoformat(),
            'raw_file': raw_filename,
            'processed_file': processed_filename,
            'frames_processed': len(df_processed),
            'total_frames': len(df_raw)
        }
        
    except Exception as e:
        processing_status[task_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error durante el procesamiento: {str(e)}',
            'end_time': datetime.now().isoformat()
        }

@app.route('/')
def index():
    """Página principal."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint para subir archivos de video."""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No se encontró el archivo de video'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Tipo de archivo no permitido'}), 400
        
        # Generar nombre único para el archivo
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Guardar archivo
        file.save(file_path)
        
        # Obtener parámetros opcionales
        smooth_enabled = request.form.get('smooth_enabled', 'true').lower() == 'true'
        fix_legs = request.form.get('fix_legs', 'true').lower() == 'true'
        
        # Generar ID único para el procesamiento
        task_id = str(uuid.uuid4())
        
        # Iniciar procesamiento en hilo separado
        thread = threading.Thread(
            target=process_video_async,
            args=(file_path, task_id, smooth_enabled, fix_legs)
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'Archivo subido exitosamente. Procesamiento iniciado.'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error al subir archivo: {str(e)}'}), 500

@app.route('/status/<task_id>')
def get_status(task_id):
    """Obtiene el estado del procesamiento."""
    if task_id not in processing_status:
        return jsonify({'error': 'ID de tarea no encontrado'}), 404
    
    return jsonify(processing_status[task_id])

@app.route('/download/<filename>')
def download_file(filename):
    """Descarga archivos procesados."""
    try:
        file_path = os.path.join(RESULTS_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'Archivo no encontrado'}), 404
    except Exception as e:
        return jsonify({'error': f'Error al descargar archivo: {str(e)}'}), 500

@app.route('/results')
def list_results():
    """Lista todos los archivos de resultados disponibles."""
    try:
        files = []
        for filename in os.listdir(RESULTS_FOLDER):
            if filename.endswith('.csv'):
                file_path = os.path.join(RESULTS_FOLDER, filename)
                file_stat = os.stat(file_path)
                files.append({
                    'filename': filename,
                    'size': file_stat.st_size,
                    'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat()
                })
        
        # Ordenar por fecha de creación (más recientes primero)
        files.sort(key=lambda x: x['created'], reverse=True)
        return jsonify({'files': files})
        
    except Exception as e:
        return jsonify({'error': f'Error al listar archivos: {str(e)}'}), 500

@app.route('/health')
def health_check():
    """Endpoint de verificación de salud."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'upload_folder': UPLOAD_FOLDER,
        'results_folder': RESULTS_FOLDER
    })

if __name__ == '__main__':
    print("=== SISTEMA DE PROCESAMIENTO DE VIDEOS ===")
    print(f"Directorio de subidas: {UPLOAD_FOLDER}")
    print(f"Directorio de resultados: {RESULTS_FOLDER}")
    print("Servidor iniciando en http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
