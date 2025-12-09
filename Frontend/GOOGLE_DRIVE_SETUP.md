# Configuración de Google Drive API

Para habilitar la funcionalidad de subir videos a Google Drive, necesitas configurar las credenciales de Google Drive API.

## Opción 1: Service Account (Recomendado para servidor)

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto o selecciona uno existente
3. Habilita la API de Google Drive:
   - Ve a "APIs & Services" > "Library"
   - Busca "Google Drive API" y habilítala
4. Crea una Service Account:
   - Ve a "APIs & Services" > "Credentials"
   - Haz clic en "Create Credentials" > "Service Account"
   - Completa el formulario y crea la cuenta
5. Descarga el archivo JSON de credenciales:
   - En la lista de Service Accounts, haz clic en la cuenta creada
   - Ve a la pestaña "Keys"
   - Haz clic en "Add Key" > "Create new key"
   - Selecciona "JSON" y descarga el archivo
6. Renombra el archivo descargado a `credentials.json` y colócalo en el directorio `Frontend/`
7. Comparte la carpeta de Google Drive con el email de la Service Account:
   - Abre la carpeta en Google Drive: https://drive.google.com/drive/folders/1dLR_z0DHWoUfWh2egFpk2I41Ng0okqTs
   - Haz clic derecho > "Compartir"
   - Agrega el email de la Service Account (se encuentra en el archivo credentials.json, campo "client_email")
   - Dale permisos de "Editor"

## Opción 2: OAuth2 (Para desarrollo/testing)

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Crea un nuevo proyecto o selecciona uno existente
3. Habilita la API de Google Drive (igual que en Opción 1)
4. Crea credenciales OAuth2:
   - Ve a "APIs & Services" > "Credentials"
   - Haz clic en "Create Credentials" > "OAuth client ID"
   - Selecciona "Desktop app" como tipo de aplicación
   - Descarga el archivo JSON de credenciales
5. Renombra el archivo a `client_secret.json` y colócalo en el directorio `Frontend/`
6. La primera vez que uses la aplicación, se abrirá un navegador para autenticarte
7. Después de autenticarte, se creará automáticamente un archivo `token.json` con tus credenciales

## Instalación de dependencias

Instala las librerías necesarias:

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

## Verificación

Una vez configurado, cuando proceses un video y marques la opción "Guardar en Google Drive", el video se subirá automáticamente a la carpeta compartida.

Si hay errores, revisa los logs del servidor para más detalles.

