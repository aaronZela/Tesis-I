// Variables globales
let currentTaskId = null;
let statusInterval = null;
let startTime = null;

// Inicialización cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    setupFileUpload();
    setupFormSubmission();
    loadFilesList();
    
    // Cargar lista de archivos cada 30 segundos
    setInterval(loadFilesList, 30000);
}

// Configurar drag and drop para subida de archivos
function setupFileUpload() {
    const uploadArea = document.getElementById('fileUploadArea');
    const fileInput = document.getElementById('videoFile');
    
    // Prevenir comportamiento por defecto del navegador
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Resaltar área de drop
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    // Manejar archivos soltados
    uploadArea.addEventListener('drop', handleDrop, false);
    
    // Manejar clic en área de upload
    uploadArea.addEventListener('click', () => fileInput.click());
    
    // Mostrar nombre del archivo seleccionado
    fileInput.addEventListener('change', handleFileSelect);
}

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function highlight(e) {
    document.getElementById('fileUploadArea').classList.add('dragover');
}

function unhighlight(e) {
    document.getElementById('fileUploadArea').classList.remove('dragover');
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    
    if (files.length > 0) {
        document.getElementById('videoFile').files = files;
        handleFileSelect();
    }
}

function handleFileSelect() {
    const fileInput = document.getElementById('videoFile');
    const file = fileInput.files[0];
    
    if (file) {
        const uploadContent = document.querySelector('.upload-content');
        uploadContent.innerHTML = `
            <i class="fas fa-file-video upload-icon" style="color: #28a745;"></i>
            <p><strong>${file.name}</strong></p>
            <p class="file-types">Tamaño: ${formatFileSize(file.size)}</p>
            <p class="file-types">Haz clic para cambiar archivo</p>
        `;
    }
}

// Configurar envío del formulario
function setupFormSubmission() {
    const form = document.getElementById('uploadForm');
    form.addEventListener('submit', handleFormSubmit);
}

async function handleFormSubmit(e) {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const uploadBtn = document.getElementById('uploadBtn');
    
    // Validar archivo
    const fileInput = document.getElementById('videoFile');
    if (!fileInput.files[0]) {
        showToast('Por favor selecciona un archivo de video', 'error');
        return;
    }
    
    // Mostrar loading
    showLoading(true);
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Subiendo...';
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok && result.success) {
            currentTaskId = result.task_id;
            startTime = Date.now();
            showToast('Archivo subido exitosamente. Iniciando procesamiento...', 'success');
            
            // Mostrar sección de progreso
            document.getElementById('progressSection').style.display = 'block';
            document.getElementById('progressSection').scrollIntoView({ behavior: 'smooth' });
            
            // Iniciar monitoreo de estado
            startStatusMonitoring();
        } else {
            showToast(result.error || 'Error al subir archivo', 'error');
        }
    } catch (error) {
        console.error('Error:', error);
        showToast('Error de conexión. Por favor intenta de nuevo.', 'error');
    } finally {
        showLoading(false);
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = '<i class="fas fa-play"></i> Procesar Video';
    }
}

// Monitorear estado del procesamiento
function startStatusMonitoring() {
    if (statusInterval) {
        clearInterval(statusInterval);
    }
    
    statusInterval = setInterval(checkStatus, 1000); // Verificar cada segundo
}

function stopStatusMonitoring() {
    if (statusInterval) {
        clearInterval(statusInterval);
        statusInterval = null;
    }
}

async function checkStatus() {
    if (!currentTaskId) return;
    
    try {
        const response = await fetch(`/status/${currentTaskId}`);
        const status = await response.json();
        
        if (response.ok) {
            updateProgress(status);
            
            if (status.status === 'completed') {
                stopStatusMonitoring();
                showResults(status);
                showToast('¡Procesamiento completado exitosamente!', 'success');
                loadFilesList(); // Actualizar lista de archivos
            } else if (status.status === 'error') {
                stopStatusMonitoring();
                showToast(`Error durante el procesamiento: ${status.message}`, 'error');
                hideProgress();
            }
        } else {
            console.error('Error al obtener estado:', status.error);
        }
    } catch (error) {
        console.error('Error al verificar estado:', error);
    }
}

// Actualizar barra de progreso
function updateProgress(status) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const statusValue = document.getElementById('statusValue');
    const timeValue = document.getElementById('timeValue');
    
    // Actualizar progreso
    const progress = status.progress || 0;
    progressFill.style.width = `${progress}%`;
    progressText.textContent = status.message || 'Procesando...';
    statusValue.textContent = status.status === 'processing' ? 'Procesando' : status.status;
    
    // Actualizar tiempo transcurrido
    if (startTime) {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        timeValue.textContent = formatTime(elapsed);
    }
}

// Mostrar resultados
function showResults(status) {
    const resultsSection = document.getElementById('resultsSection');
    const resultsInfo = document.getElementById('resultsInfo');
    const downloadButtons = document.getElementById('downloadButtons');
    
    // Información de resultados
    resultsInfo.innerHTML = `
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 15px;">
            <i class="fas fa-check-circle" style="color: #28a745; font-size: 2rem;"></i>
            <div>
                <h3 style="margin: 0; color: #28a745;">Procesamiento Completado</h3>
                <p style="margin: 5px 0; color: #666;">${status.message}</p>
            </div>
        </div>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
            <div>
                <strong>Frames procesados:</strong> ${status.frames_processed || 'N/A'}
            </div>
            <div>
                <strong>Total de frames:</strong> ${status.total_frames || 'N/A'}
            </div>
            <div>
                <strong>Archivos generados:</strong> 2 (Raw y Procesado)
            </div>
        </div>
    `;
    
    // Botones de descarga
    downloadButtons.innerHTML = `
        <button class="btn btn-success" onclick="downloadFile('${status.raw_file}')">
            <i class="fas fa-download"></i> Descargar CSV Raw
        </button>
        <button class="btn btn-success" onclick="downloadFile('${status.processed_file}')">
            <i class="fas fa-download"></i> Descargar CSV Procesado
        </button>
    `;
    
    // Mostrar sección de resultados
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
    
    // Ocultar sección de progreso
    hideProgress();
}

// Ocultar sección de progreso
function hideProgress() {
    const progressSection = document.getElementById('progressSection');
    progressSection.style.display = 'none';
    
    // Resetear valores
    document.getElementById('progressFill').style.width = '0%';
    document.getElementById('progressText').textContent = 'Iniciando...';
    document.getElementById('statusValue').textContent = 'Preparando';
    document.getElementById('timeValue').textContent = '00:00';
    
    currentTaskId = null;
    startTime = null;
}

// Descargar archivo
async function downloadFile(filename) {
    try {
        const response = await fetch(`/download/${filename}`);
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            
            showToast(`Archivo ${filename} descargado exitosamente`, 'success');
        } else {
            const error = await response.json();
            showToast(`Error al descargar archivo: ${error.error}`, 'error');
        }
    } catch (error) {
        console.error('Error al descargar:', error);
        showToast('Error al descargar archivo', 'error');
    }
}

// Cargar lista de archivos
async function loadFilesList() {
    try {
        const response = await fetch('/results');
        const data = await response.json();
        
        if (response.ok) {
            displayFilesList(data.files);
        } else {
            console.error('Error al cargar archivos:', data.error);
        }
    } catch (error) {
        console.error('Error al cargar lista de archivos:', error);
    }
}

// Mostrar lista de archivos
function displayFilesList(files) {
    const filesList = document.getElementById('filesList');
    
    if (files.length === 0) {
        filesList.innerHTML = `
            <div class="text-center" style="padding: 40px; color: #666;">
                <i class="fas fa-folder-open" style="font-size: 3rem; margin-bottom: 15px; opacity: 0.5;"></i>
                <p>No hay archivos procesados aún</p>
                <p style="font-size: 0.9rem;">Sube tu primer video para comenzar</p>
            </div>
        `;
        return;
    }
    
    filesList.innerHTML = files.map(file => `
        <div class="file-item">
            <div class="file-info">
                <div class="file-name">${file.filename}</div>
                <div class="file-meta">
                    Tamaño: ${formatFileSize(file.size)} | 
                    Creado: ${formatDate(file.created)}
                </div>
            </div>
            <div class="file-actions">
                <button class="btn btn-success" onclick="downloadFile('${file.filename}')">
                    <i class="fas fa-download"></i>
                </button>
            </div>
        </div>
    `).join('');
}

// Actualizar lista de archivos (función global para botón)
function refreshFiles() {
    loadFilesList();
    showToast('Lista de archivos actualizada', 'success');
}

// Mostrar/ocultar loading overlay
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = show ? 'flex' : 'none';
}

// Mostrar notificaciones toast
function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <i class="fas fa-${getToastIcon(type)}"></i>
            <span>${message}</span>
        </div>
    `;
    
    container.appendChild(toast);
    
    // Auto-remover después de 5 segundos
    setTimeout(() => {
        toast.style.animation = 'slideOut 0.3s ease forwards';
        setTimeout(() => {
            if (container.contains(toast)) {
                container.removeChild(toast);
            }
        }, 300);
    }, 5000);
}

function getToastIcon(type) {
    const icons = {
        success: 'check-circle',
        error: 'exclamation-circle',
        warning: 'exclamation-triangle',
        info: 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// Funciones de utilidad
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString('es-ES', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatTime(seconds) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Limpiar intervalos cuando se cierra la página
window.addEventListener('beforeunload', function() {
    if (statusInterval) {
        clearInterval(statusInterval);
    }
});
