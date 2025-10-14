"""
Configurações centralizadas do projeto Analyse Doc.
"""
import os
from typing import Dict, List
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Configurações gerais da aplicação."""
    APP_TITLE = "Analyse Doc - Analise documentos com IA"
    APP_ICON = "📑"
    LAYOUT = "wide"
    
    # Limites de processamento
    MAX_FILE_SIZE_MB = 50
    MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
    SMALL_DOCUMENT_THRESHOLD = 25000  # caracteres
    
    # Configurações de chunking
    DEFAULT_CHUNK_SIZE = 2000
    DEFAULT_CHUNK_OVERLAP = 200
    MIN_CHUNK_SIZE = 1000
    MAX_CHUNK_SIZE = 4000
    
    # Configurações de recuperação
    DEFAULT_K_CHUNKS = 2
    MIN_K_CHUNKS = 1
    MAX_K_CHUNKS = 5
    
    # Cache
    ENABLE_CACHE = True
    CACHE_DIR = ".cache"
    
    # Configurações de retry
    MAX_RETRIES = 5
    RETRY_DELAY = 3


@dataclass
class ModelConfig:
    """Configurações de modelos de IA."""
    PROVIDERS = {
        'Groq': {
            'modelos': [
                'llama-3.3-70b-versatile',
                'llama-3.1-8b-instant',
                'mixtral-8x7b-32768',
                'gemma2-9b-it'
            ],
            'temperatura_padrao': 0.7,
            'max_tokens': 4096
        },
        'OpenAI': {
            'modelos': [
                'gpt-4o-mini',
                'gpt-4o',
                'gpt-4-turbo',
                'gpt-3.5-turbo'
            ],
            'temperatura_padrao': 0.7,
            'max_tokens': 4096
        }
    }


@dataclass
class FileTypes:
    """Tipos de arquivos suportados."""
    SUPPORTED_TYPES = ['Site', 'Youtube', 'Pdf', 'Docx', 'Csv', 'Txt']
    
    FILE_EXTENSIONS = {
        'Pdf': ['.pdf'],
        'Docx': ['.docx', '.doc'],
        'Csv': ['.csv'],
        'Txt': ['.txt']
    }
    
    MIME_TYPES = {
        'Pdf': 'application/pdf',
        'Docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'Csv': 'text/csv',
        'Txt': 'text/plain'
    }


# Stopwords em português para recuperação de chunks
STOPWORDS_PT = {
    'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da', 'dos', 'das',
    'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com', 'sem', 'sob', 'sobre',
    'e', 'ou', 'mas', 'que', 'porque', 'quando', 'onde', 'como', 'qual', 'quais',
    'é', 'são', 'foi', 'eram', 'ao', 'aos', 'à', 'às', 'pelo', 'pela', 'pelos', 'pelas',
    'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas', 'aquele', 'aquela',
    'the', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'
}

# Estilos CSS customizados
CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #4F8BF9;
        text-align: center;
        margin-bottom: 1rem;
    }
    .chat-message-ai {
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
        background-color: rgba(100, 149, 237, 0.1);
        border-left: 3px solid #4F8BF9;
        animation: fadeIn 0.3s;
    }
    .chat-message-human {
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
        background-color: rgba(220, 220, 220, 0.2);
        border-left: 3px solid #808080;
        animation: fadeIn 0.3s;
    }
    .stButton > button {
        background-color: #4F8BF9;
        color: white;
        font-weight: 500;
        border-radius: 0.3rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #3A66CC;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: rgba(79, 139, 249, 0.05);
        border: 1px solid rgba(79, 139, 249, 0.2);
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .success-message {
        color: #28a745;
        font-weight: 500;
    }
    .error-message {
        color: #dc3545;
        font-weight: 500;
    }
    .warning-message {
        color: #ffc107;
        font-weight: 500;
    }
</style>
"""
