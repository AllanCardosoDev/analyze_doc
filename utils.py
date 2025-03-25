import logging
from cryptography.fernet import Fernet
import re

def setup_logging():
    """Configura o logging para a aplicação."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='app.log',
        filemode='a'
    )

def sanitize_input(input_text):
    """Sanitiza a entrada do usuário para prevenir injeções maliciosas."""
    # Remove caracteres especiais e tags HTML
    sanitized = re.sub(r'[<>&;]', '', input_text)
    return sanitized

def encrypt_data(data):
    """Encripta dados sensíveis."""
    key = Fernet.generate_key()
    fernet = Fernet(key)
    return fernet.encrypt(data.encode())

def decrypt_data(encrypted_data, key):
    """Decripta dados encriptados."""
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_data).decode()
