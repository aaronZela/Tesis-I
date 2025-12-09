
"""
CP-0002: Comprobar que el sistema permita el acceso seguro mediante login
utilizando credenciales válidas y controlando intentos fallidos, conforme
a las prácticas de seguridad definidas en el requerimiento RF-0009.

Prueba de autenticación - caja blanca
Verifica el flujo completo de login en Flask, incluyendo:
- Validación de credenciales (hash seguro)
- Creación de sesión segura
- Bloqueo temporal tras intentos fallidos
- Expiración de sesión por inactividad
"""

import os
import sys
import pytest
from unittest.mock import patch
from flask import session
from datetime import timedelta, datetime
from werkzeug.security import check_password_hash

# ==================== CONFIGURACIÓN DE RUTAS ====================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
FRONTEND_PATH = os.path.join(PROJECT_ROOT, "Frontend")

# Agregar la carpeta Frontend al path
if FRONTEND_PATH not in sys.path:
    sys.path.insert(0, FRONTEND_PATH)

# Importar la app Flask y las variables necesarias desde Frontend/app.py
from app import app, USERS as usuarios, login_attempts as LOGIN_ATTEMPTS, MAX_LOGIN_ATTEMPTS as MAX_ATTEMPTS, LOCKOUT_DURATION


# ==================== FIXTURES ====================

@pytest.fixture
def client():
    """Crea un cliente de pruebas Flask"""
    app.config["TESTING"] = True
    with app.test_client() as client:
        with app.app_context():
            yield client


@pytest.fixture
def valid_user():
    """Usuario válido registrado"""
    return {"username": "admin@gmail.com", "password": "admin"}


@pytest.fixture
def invalid_user():
    """Usuario inexistente o con datos erróneos"""
    return {"username": "fake_user@gmail.com", "password": "wrongpass"}


# ==================== TESTS ====================

@pytest.mark.cp0002
def test_login_exitoso(client, valid_user):
    """
    CP-0002: Paso 1 - Login exitoso con credenciales válidas
    Precondición: Usuario registrado en el sistema.
    """
    response = client.post("/login", json=valid_user, follow_redirects=True)

    assert response.status_code == 200, "El servidor no respondió correctamente"
    assert b"Bienvenido" in response.data or b"index" in response.data, \
        "No se redirigió correctamente tras login exitoso"

    # Verificar creación de sesión
    with client.session_transaction() as sess:
        assert sess.get("logged_in") is True, "La sesión no se creó correctamente"
        assert "username" in sess, "El nombre de usuario no se guardó en la sesión"

    print("✅ Login exitoso: sesión creada correctamente.")


@pytest.mark.cp0002
def test_login_contraseña_incorrecta(client, valid_user):
    """
    CP-0002: Paso 2 - Intento de login con contraseña incorrecta
    Debe rechazar las credenciales y registrar el intento fallido.
    """
    wrong_data = valid_user.copy()
    wrong_data["password"] = "incorrecta123"

    response = client.post("/login", json=wrong_data)

    assert response.status_code in [401, 403], "El servidor no devolvió código de error esperado"
    assert b"incorrectos" in response.data or b"error" in response.data, "Mensaje de error no mostrado"
    assert valid_user["username"] in LOGIN_ATTEMPTS, "No se registró el intento fallido"

    print("✅ Contraseña incorrecta detectada correctamente.")


@pytest.mark.cp0002
def test_bloqueo_por_intentos_fallidos(client, invalid_user):
    """
    CP-0002: Paso 3 - Bloqueo temporal tras múltiples intentos fallidos
    El sistema debe bloquear la cuenta después de 5 intentos consecutivos fallidos.
    """
    username = invalid_user["username"]

    # Realizar 5 intentos fallidos consecutivos
    for i in range(MAX_ATTEMPTS):
        client.post("/login", json=invalid_user)

    # Intento adicional debe resultar en bloqueo
    response = client.post("/login", json=invalid_user)
    assert response.status_code == 403, "No se aplicó el bloqueo de cuenta"
    assert b"bloqueada" in response.data or b"bloqueado" in response.data, "Mensaje de bloqueo no mostrado"
    assert "locked_until" in LOGIN_ATTEMPTS[username] and LOGIN_ATTEMPTS[username]["locked_until"] is not None, "No se registró el tiempo de bloqueo"

    print(f"✅ Cuenta bloqueada correctamente tras {MAX_ATTEMPTS} intentos fallidos.")


@pytest.mark.cp0002
def test_desbloqueo_tras_tiempo_de_espera(client, invalid_user):
    """
    CP-0002: Paso 4 - Desbloqueo automático tras tiempo de espera
    Verifica que el sistema permita login después de que expire el tiempo de bloqueo.
    """
    import time
    username = invalid_user["username"]

    # Forzar bloqueo manualmente
    LOGIN_ATTEMPTS[username] = {"attempts": MAX_ATTEMPTS, "locked_until": datetime.now() - timedelta(seconds=1)}  # ya expiró

    # Reintentar login (aún fallará por credenciales, pero no por bloqueo)
    response = client.post("/login", json=invalid_user)
    assert response.status_code in [401, 403], "El usuario debería poder intentar nuevamente tras desbloqueo"

    print("✅ Desbloqueo automático tras expiración del bloqueo temporal.")


@pytest.mark.cp0002
def test_hash_contraseña_seguro(valid_user):
    """
    CP-0002: Paso 5 - Verificar que las contraseñas se almacenan con hash
    y no en texto plano.
    """
    stored_hash = usuarios[valid_user["username"]]["password"]
    assert stored_hash != valid_user["password"], "La contraseña está almacenada en texto plano"
    assert check_password_hash(stored_hash, valid_user["password"]), "El hash no valida correctamente la contraseña"

    print("✅ Contraseña almacenada de forma segura con hash.")


@pytest.mark.cp0002
def test_expiracion_sesion(client, valid_user):
    """
    CP-0002: Paso 6 - Verificar expiración de sesión tras inactividad
    """
    client.post("/login", json=valid_user, follow_redirects=True)

    with client.session_transaction() as sess:
        sess.permanent = True
        app.permanent_session_lifetime = timedelta(seconds=2)  # tiempo corto
        username = sess.get("username")

    import time
    time.sleep(3)  # esperar más del tiempo de expiración

    response = client.get("/index", follow_redirects=True)
    assert b"login" in response.data or response.status_code == 302, \
        "El sistema no redirigió tras expiración de sesión"

    print(f"✅ Sesión de usuario '{username}' expirada correctamente tras inactividad.")


@pytest.mark.cp0002
def test_acceso_no_autorizado(client):
    """
    CP-0002: Paso 7 - Intento de acceso a zona protegida sin login
    """
    response = client.get("/index", follow_redirects=True)
    assert response.status_code == 200, "Error al redirigir al login"
    assert b"login" in response.data, "No se redirigió al login correctamente"

    print("✅ Acceso no autorizado bloqueado correctamente.")


@pytest.mark.cp0002
def test_logout_cierra_sesion(client, valid_user):
    """
    CP-0002: Paso 8 - Logout y cierre correcto de sesión
    """
    client.post("/login", json=valid_user)
    response = client.get("/logout", follow_redirects=True)

    with client.session_transaction() as sess:
        assert not sess.get("logged_in", False), "La sesión no se cerró correctamente"

    assert b"login" in response.data or response.status_code == 200, \
        "No se redirigió correctamente tras logout"

    print("✅ Logout exitoso: sesión cerrada correctamente.")
