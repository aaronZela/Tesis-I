# Pruebas de Integración - CP-0001 y CP-0002

## Guía de Instalación

### Requisitos Previos

1. **Python 3.8+** instalado
2. **Entorno virtual** activado (recomendado)

### Instalación de Dependencias

#### 1. Activar el entorno virtual (si existe)

**PowerShell:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

**CMD (Símbolo del sistema):**
```cmd
venv\Scripts\activate.bat
```

#### 2. Instalar PyTest y dependencias

```bash
python -m pip install --upgrade pip
pip install pytest pytest-mock pandas numpy torch scikit-learn
```

**Nota:** Si ya tienes un `requirements.txt`, también puedes instalar desde ahí:
```bash
pip install -r requirements.txt
```

#### 3. Verificar instalación

```bash
pytest --version
python -c "import torch, pandas, numpy, sklearn; print('✅ Todas las dependencias instaladas')"
```

---

## CP-0001: Prueba de Extracción de Pose

**Objetivo:** Verificar detección correcta de puntos clave del cuerpo humano  
**Tipo:** Prueba de caja negra  
**Servicio:** Extracción

### Requisitos
- MediaPipe instalado (`pip install mediapipe opencv-python`)
- Archivos de video `.mp4` en `Backend/BD/` o `BD/` (raíz del proyecto)

### Ejecución

```bash
# Automático: busca videos en Backend/BD o BD/
pytest -q Pruebas/test_extraccion_cp0001.py -m cp0001

# Con video específico (opcional)
$env:TEST_VIDEO_MP4="D:\ruta\al\video.mp4"
pytest -q Pruebas/test_extraccion_cp0001.py -m cp0001
```

### Criterios Verificados
- ✅ Se generan CSVs y no están vacíos
- ✅ Columnas esperadas de 32 landmarks (x,y,z,visibility)
- ✅ Coordenadas procesadas en rango [0,1]
- ✅ Estabilidad: >= 85% de puntos válidos por frame en promedio
- ✅ Frames con < 85% puntos se descartan
- ✅ Formato CSV válido

---

## CP-0002: Prueba de Integración con Modelos IA

**Objetivo:** Comprobar que los datos de MediaPipe sean correctamente procesados por los modelos IA  
**Tipo:** Prueba de integración (caja blanca)  
**Servicio:** Integración entre Extracción y Entrenamiento

### Requisitos
- Archivos CSV procesados en `Backend/Coordenadas_csv/` (terminan en `_processed.csv`)
- PyTorch instalado (`pip install torch`)
- Modelos de IA disponibles (`CVAE-LSTM1.py`, etc.)

### Ejecución

```bash
# Ejecutar todas las pruebas CP-0002
pytest -q Pruebas/test_integracion_ia_cp0002.py -m cp0002

# Ejecutar una prueba específica
pytest -q Pruebas/test_integracion_ia_cp0002.py::test_carga_csv_y_validacion_estructura -m cp0002
```

### Pruebas Incluidas

1. **test_carga_csv_y_validacion_estructura**: Carga CSV y valida estructura
2. **test_integracion_stepdataset_normal**: Integración con StepDataset
3. **test_manejo_valores_nan**: Manejo de datos con NaN (Sub-variación 1)
4. **test_secuencias_longitud_variable**: Longitudes muy variables (Sub-variación 2)
5. **test_coordenadas_multiples_videos**: Múltiples videos (Sub-variación 3)
6. **test_inicio_entrenamiento_sin_excepciones**: Inicio de entrenamiento sin errores
7. **test_rechazo_datos_formato_incorrecto**: Rechazo de formato inválido (Condición de Fallo)
8. **test_normalizacion_datos_correcta**: Verificación de normalización
9. **test_validacion_completa_pipeline**: Validación completa del pipeline

### Criterios Verificados

**Condición de Éxito:**
- ✅ Los modelos IA reciben datos sin errores
- ✅ Procede al entrenamiento o generación correcta
- ✅ Normalización correcta de datos
- ✅ Estructura de datos válida

**Condición de Fallo (debe detectar):**
- ❌ Error de formato en datos
- ❌ Fallo en normalización
- ❌ Rechazo de datos por el modelo
- ❌ Extension 1.1: Registrar error si formato no es válido

**Sub-variaciones:**
- ✅ Datos con valores NaN
- ✅ Secuencia de longitudes muy variables
- ✅ Coordenadas de múltiples videos

---

## Ejecutar Todas las Pruebas

```bash
# Todas las pruebas
pytest -q Pruebas/

# Solo CP-0001
pytest -q Pruebas/test_extraccion_cp0001.py -m cp0001

# Solo CP-0002
pytest -q Pruebas/test_integracion_ia_cp0002.py -m cp0002

# Con reporte detallado
pytest -v Pruebas/

# Con cobertura (si está instalado pytest-cov)
pytest --cov=Backend Pruebas/
```

---

## Solución de Problemas

### Error: "No se encontraron archivos CSV"
- Verifica que `Backend/Coordenadas_csv/` contenga archivos `*_processed.csv`
- Ejecuta primero el servicio de extracción para generar CSV

### Error: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch
```

### Error: "PytestUnknownMarkWarning"
- Verifica que `pytest.ini` esté en la raíz del proyecto o en `Pruebas/`
- O ejecuta sin el marcador: `pytest Pruebas/test_*.py` (sin `-m`)

### Error: "ExecutionPolicy" en PowerShell
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

---

## Notas

- Las pruebas usan **unittest.mock** (equivalente a Mockito de Java) para simulación
- **PyTest** es el framework de pruebas (equivalente a JUnit de Java)
- Los fixtures (`@pytest.fixture`) preparan datos de prueba automáticamente
- Las marcas `@pytest.mark.cp0001` y `@pytest.mark.cp0002` permiten filtrar pruebas
