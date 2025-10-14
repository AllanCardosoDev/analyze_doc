"""
Funções utilitárias para o projeto Analyse Doc.
"""
import hashlib
import re
import os
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse
import streamlit as st

logger = logging.getLogger(__name__)


def validate_url(url: str) -> bool:
    """
    Valida se uma URL está bem formada.
    
    Args:
        url: URL para validar
        
    Returns:
        bool: True se válida, False caso contrário
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_youtube_url(url: str) -> Optional[str]:
    """
    Valida e extrai o ID de um vídeo do YouTube.
    
    Args:
        url: URL do vídeo do YouTube
        
    Returns:
        str: ID do vídeo ou None se inválido
    """
    youtube_patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    
    for pattern in youtube_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return None


def calculate_file_hash(content: str) -> str:
    """
    Calcula o hash MD5 de um conteúdo.
    
    Args:
        content: Conteúdo para calcular hash
        
    Returns:
        str: Hash MD5 em hexadecimal
    """
    return hashlib.md5(content.encode('utf-8')).hexdigest()


def format_file_size(size_bytes: int) -> str:
    """
    Formata tamanho de arquivo em formato legível.
    
    Args:
        size_bytes: Tamanho em bytes
        
    Returns:
        str: Tamanho formatado (ex: "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def estimate_tokens(text: str) -> int:
    """
    Estima o número de tokens em um texto.
    Aproximação: 1 token ≈ 4 caracteres para português/inglês.
    
    Args:
        text: Texto para estimar
        
    Returns:
        int: Número estimado de tokens
    """
    return len(text) // 4


def estimate_cost(tokens: int, provider: str, model: str) -> Dict[str, float]:
    """
    Estima o custo de uso de tokens.
    
    Args:
        tokens: Número de tokens
        provider: Provedor (Groq, OpenAI)
        model: Nome do modelo
        
    Returns:
        dict: Custo estimado de input e output
    """
    # Custos aproximados por 1M tokens (atualizar com valores reais)
    pricing = {
        'OpenAI': {
            'gpt-4o': {'input': 2.50, 'output': 10.00},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
            'gpt-4-turbo': {'input': 10.00, 'output': 30.00},
            'gpt-3.5-turbo': {'input': 0.50, 'output': 1.50}
        },
        'Groq': {
            'default': {'input': 0.10, 'output': 0.10}  # Groq é geralmente mais barato
        }
    }
    
    if provider == 'Groq':
        rates = pricing['Groq']['default']
    else:
        rates = pricing.get(provider, {}).get(model, {'input': 0, 'output': 0})
    
    cost_input = (tokens / 1_000_000) * rates['input']
    cost_output = (tokens / 1_000_000) * rates['output']
    
    return {
        'input': cost_input,
        'output': cost_output,
        'total_estimated': cost_input + cost_output
    }


def sanitize_filename(filename: str) -> str:
    """
    Remove caracteres inválidos de um nome de arquivo.
    
    Args:
        filename: Nome do arquivo original
        
    Returns:
        str: Nome do arquivo sanitizado
    """
    # Remove caracteres especiais mantendo apenas alfanuméricos, underscores e pontos
    sanitized = re.sub(r'[^\w\s.-]', '', filename)
    # Remove espaços múltiplos
    sanitized = re.sub(r'\s+', '_', sanitized)
    return sanitized


def create_cache_key(content: str, tipo: str) -> str:
    """
    Cria uma chave única para cache baseada no conteúdo e tipo.
    
    Args:
        content: Conteúdo do documento
        tipo: Tipo do documento
        
    Returns:
        str: Chave de cache
    """
    content_hash = calculate_file_hash(content)
    return f"{tipo}_{content_hash}"


def setup_logging(level: str = "INFO") -> None:
    """
    Configura o sistema de logging.
    
    Args:
        level: Nível de log (DEBUG, INFO, WARNING, ERROR)
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('analyse_doc.log', encoding='utf-8')
        ]
    )


def safe_session_state_get(key: str, default: Any = None) -> Any:
    """
    Obtém valor do session_state de forma segura.
    
    Args:
        key: Chave do session_state
        default: Valor padrão se não existir
        
    Returns:
        Valor do session_state ou default
    """
    return st.session_state.get(key, default)


def safe_session_state_set(key: str, value: Any) -> None:
    """
    Define valor no session_state de forma segura.
    
    Args:
        key: Chave do session_state
        value: Valor a ser definido
    """
    st.session_state[key] = value


def clear_session_state_prefix(prefix: str) -> None:
    """
    Limpa todas as chaves do session_state que começam com um prefixo.
    
    Args:
        prefix: Prefixo das chaves a serem removidas
    """
    keys_to_remove = [key for key in st.session_state.keys() if key.startswith(prefix)]
    for key in keys_to_remove:
        del st.session_state[key]


def validate_api_key(api_key: str, provider: str) -> tuple[bool, str]:
    """
    Valida o formato de uma API key.
    
    Args:
        api_key: Chave API para validar
        provider: Provedor (Groq, OpenAI)
        
    Returns:
        tuple: (válido, mensagem)
    """
    if not api_key or len(api_key) < 10:
        return False, "API key muito curta ou vazia"
    
    if provider == 'OpenAI':
        if not api_key.startswith('sk-'):
            return False, "API key da OpenAI deve começar com 'sk-'"
        if len(api_key) < 40:
            return False, "API key da OpenAI parece inválida (muito curta)"
    
    elif provider == 'Groq':
        if not api_key.startswith('gsk_'):
            return False, "API key da Groq deve começar com 'gsk_'"
    
    return True, "API key válida"


def format_document_info(info: Dict[str, Any]) -> str:
    """
    Formata informações do documento para exibição.
    
    Args:
        info: Dicionário com informações do documento
        
    Returns:
        str: Informações formatadas em HTML
    """
    size_formatted = format_file_size(info.get('tamanho', 0))
    tokens_estimated = estimate_tokens(str(info.get('tamanho', 0)))
    
    return f"""
    <div class='info-box'>
        <strong>📄 Documento Carregado</strong><br>
        <small>
        • Tipo: {info.get('tipo', 'Desconhecido')}<br>
        • Tamanho: {size_formatted} ({info.get('tamanho', 0):,} caracteres)<br>
        • Páginas estimadas: {info.get('num_paginas', 0)}<br>
        • Chunks processados: {info.get('num_chunks', 0)}<br>
        • Tokens estimados: ~{tokens_estimated:,}
        </small>
    </div>
    """
