// Toggle password visibility
document.getElementById('togglePassword').addEventListener('click', function() {
    const passwordInput = document.getElementById('password');
    const icon = this;
    
    if (passwordInput.type === 'password') {
        passwordInput.type = 'text';
        icon.classList.remove('fa-eye');
        icon.classList.add('fa-eye-slash');
    } else {
        passwordInput.type = 'password';
        icon.classList.remove('fa-eye-slash');
        icon.classList.add('fa-eye');
    }
});

// Show alert
function showAlert(message, type = 'error') {
    const alertBox = document.getElementById('alertBox');
    alertBox.className = `alert alert-${type} show`;
    alertBox.textContent = message;
    
    setTimeout(() => {
        alertBox.classList.remove('show');
    }, 5000);
}

// Show/hide loading
function showLoading(show) {
    const overlay = document.getElementById('loadingOverlay');
    overlay.classList.toggle('show', show);
}

// Handle form submission
document.getElementById('loginForm').addEventListener('submit', async function(e) {
    e.preventDefault();

    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value;
    const loginBtn = document.getElementById('loginBtn');

    if (!username) {
        showAlert('Por favor ingresa tu usuario o correo electrónico');
        return;
    }

    if (!password) {
        showAlert('Por favor ingresa tu contraseña');
        return;
    }

    loginBtn.disabled = true;
    loginBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Iniciando sesión...';
    showLoading(true);

    try {
        const response = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        });

        const data = await response.json();

        if (response.ok && data.success) {
            showAlert('¡Login exitoso! Redirigiendo...', 'success');
            setTimeout(() => window.location.href = data.redirect, 1000);
        } else {
            let errorMessage = data.error || 'Error al iniciar sesión';
            if (data.attempts_left !== undefined && data.attempts_left > 0) {
                errorMessage += ` (${data.attempts_left} intentos restantes)`;
            }
            showAlert(errorMessage, 'error');
            loginBtn.disabled = false;
            loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Iniciar Sesión';
            showLoading(false);
        }
    } catch (error) {
        console.error('Error:', error);
        showAlert('Error de conexión. Por favor intenta de nuevo.', 'error');
        loginBtn.disabled = false;
        loginBtn.innerHTML = '<i class="fas fa-sign-in-alt"></i> Iniciar Sesión';
        showLoading(false);
    }
});

// Limpiar alertas al escribir
['username', 'password'].forEach(id => {
    document.getElementById(id).addEventListener('input', function() {
        const alertBox = document.getElementById('alertBox');
        alertBox.classList.remove('show');
    });
});
