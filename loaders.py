"""
Módulo otimizado para carregamento de diferentes tipos de documentos.
Implementa validação robusta, tratamento de erros e cache.
"""
import os
import logging
from time import sleep
from typing import Optional, Tuple
import streamlit as st
from langchain_community.document_loaders import (
    WebBaseLoader,
    YoutubeLoader,
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader
)
from fake_useragent import UserAgent

from config import AppConfig, FileTypes
from utils import (
    validate_url,
    validate_youtube_url,
    calculate_file_hash,
    create_cache_key,
    safe_session_state_get,
    safe_session_state_set
)

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Classe para gerenciar carregamento de documentos com cache."""
    
    def __init__(self):
        self.config = AppConfig()
        self.cache = {}
        if self.config.ENABLE_CACHE:
            self._init_cache()
    
    def _init_cache(self):
        """Inicializa o sistema de cache."""
        if not os.path.exists(self.config.CACHE_DIR):
            os.makedirs(self.config.CACHE_DIR)
    
    def _get_from_cache(self, cache_key: str) -> Optional[str]:
        """
        Recupera documento do cache.
        
        Args:
            cache_key: Chave de cache
            
        Returns:
            str: Conteúdo do documento ou None
        """
        if not self.config.ENABLE_CACHE:
            return None
        
        if cache_key in self.cache:
            logger.info(f"Documento recuperado do cache em memória: {cache_key}")
            return self.cache[cache_key]
        
        cache_file = os.path.join(self.config.CACHE_DIR, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.cache[cache_key] = content
                    logger.info(f"Documento recuperado do cache em disco: {cache_key}")
                    return content
            except Exception as e:
                logger.error(f"Erro ao ler cache: {e}")
        
        return None
    
    def _save_to_cache(self, cache_key: str, content: str):
        """
        Salva documento no cache.
        
        Args:
            cache_key: Chave de cache
            content: Conteúdo a ser salvo
        """
        if not self.config.ENABLE_CACHE:
            return
        
        self.cache[cache_key] = content
        
        cache_file = os.path.join(self.config.CACHE_DIR, f"{cache_key}.txt")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Documento salvo no cache: {cache_key}")
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")


def carrega_site(url: str, use_cache: bool = True) -> Tuple[str, str]:
    """
    Carrega o conteúdo de um site web com validação e cache.
    
    Args:
        url: URL do site a ser carregado
        use_cache: Se deve usar cache
        
    Returns:
        tuple: (conteúdo, mensagem de status)
    """
    # Validar URL
    if not validate_url(url):
        error_msg = "❌ URL inválida. Por favor, forneça uma URL válida (ex: https://exemplo.com)"
        logger.error(error_msg)
        return "", error_msg
    
    # Verificar cache
    loader = DocumentLoader()
    cache_key = create_cache_key(url, "Site")
    
    if use_cache:
        cached_content = loader._get_from_cache(cache_key)
        if cached_content:
            return cached_content, "✅ Carregado do cache"
    
    documento = ''
    last_error = None
    
    for tentativa in range(AppConfig.MAX_RETRIES):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            web_loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = web_loader.load()
            documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
            
            if not documento or documento.strip() == '':
                raise ValueError("Conteúdo do site está vazio")
            
            # Salvar no cache
            loader._save_to_cache(cache_key, documento)
            
            logger.info(f"Site carregado com sucesso: {url}")
            return documento, f"✅ Site carregado ({len(documento)} caracteres)"
            
        except Exception as e:
            last_error = str(e)
            logger.warning(f"Tentativa {tentativa + 1}/{AppConfig.MAX_RETRIES} falhou: {last_error}")
            if tentativa < AppConfig.MAX_RETRIES - 1:
                sleep(AppConfig.RETRY_DELAY)
    
    error_msg = f"❌ Não foi possível carregar o site após {AppConfig.MAX_RETRIES} tentativas. Erro: {last_error}"
    logger.error(error_msg)
    return "", error_msg


def carrega_youtube(video_url: str, use_cache: bool = True) -> Tuple[str, str]:
    """
    Carrega a transcrição de um vídeo do Youtube com validação.
    
    Args:
        video_url: URL do vídeo do Youtube
        use_cache: Se deve usar cache
        
    Returns:
        tuple: (conteúdo, mensagem de status)
    """
    # Validar e extrair ID do vídeo
    video_id = validate_youtube_url(video_url)
    
    if not video_id:
        error_msg = "❌ URL do YouTube inválida. Use um formato como: https://www.youtube.com/watch?v=VIDEO_ID"
        logger.error(error_msg)
        return "", error_msg
    
    # Verificar cache
    loader = DocumentLoader()
    cache_key = create_cache_key(video_id, "Youtube")
    
    if use_cache:
        cached_content = loader._get_from_cache(cache_key)
        if cached_content:
            return cached_content, "✅ Carregado do cache"
    
    try:
        yt_loader = YoutubeLoader(
            video_id,
            add_video_info=True,
            language=['pt', 'pt-BR', 'en']
        )
        lista_documentos = yt_loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        
        if not documento or documento.strip() == '':
            raise ValueError("Não foi possível extrair a transcrição do vídeo. O vídeo pode não ter legendas disponíveis.")
        
        # Salvar no cache
        loader._save_to_cache(cache_key, documento)
        
        logger.info(f"Transcrição do YouTube carregada: {video_id}")
        return documento, f"✅ Transcrição carregada ({len(documento)} caracteres)"
        
    except Exception as e:
        error_msg = f"❌ Erro ao carregar vídeo do YouTube: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


def carrega_pdf(caminho: str, use_cache: bool = True) -> Tuple[str, str]:
    """
    Carrega o conteúdo de um arquivo PDF.
    
    Args:
        caminho: Caminho para o arquivo PDF
        use_cache: Se deve usar cache
        
    Returns:
        tuple: (conteúdo, mensagem de status)
    """
    if not os.path.exists(caminho):
        error_msg = f"❌ Arquivo não encontrado: {caminho}"
        logger.error(error_msg)
        return "", error_msg
    
    # Verificar tamanho do arquivo
    file_size = os.path.getsize(caminho)
    if file_size > AppConfig.MAX_FILE_SIZE_BYTES:
        error_msg = f"❌ Arquivo muito grande ({file_size / 1024 / 1024:.1f} MB). Limite: {AppConfig.MAX_FILE_SIZE_MB} MB"
        logger.error(error_msg)
        return "", error_msg
    
    try:
        pdf_loader = PyPDFLoader(caminho)
        lista_documentos = pdf_loader.load()
        
        # Adicionar informação de páginas
        num_paginas = len(lista_documentos)
        documento = f"Total de páginas: {num_paginas}\n\n"
        documento += '\n\n'.join([f"--- Página {i+1} ---\n{doc.page_content}" 
                                  for i, doc in enumerate(lista_documentos)])
        
        if not documento or documento.strip() == '':
            raise ValueError("O PDF parece estar vazio ou não foi possível extrair texto")
        
        logger.info(f"PDF carregado: {caminho} ({num_paginas} páginas)")
        return documento, f"✅ PDF carregado ({num_paginas} páginas, {len(documento)} caracteres)"
        
    except Exception as e:
        error_msg = f"❌ Erro ao carregar PDF: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


def carrega_docx(caminho: str, use_cache: bool = True) -> Tuple[str, str]:
    """
    Carrega o conteúdo de um arquivo Word (DOCX).
    
    Args:
        caminho: Caminho para o arquivo DOCX
        use_cache: Se deve usar cache
        
    Returns:
        tuple: (conteúdo, mensagem de status)
    """
    if not os.path.exists(caminho):
        error_msg = f"❌ Arquivo não encontrado: {caminho}"
        logger.error(error_msg)
        return "", error_msg
    
    # Verificar tamanho do arquivo
    file_size = os.path.getsize(caminho)
    if file_size > AppConfig.MAX_FILE_SIZE_BYTES:
        error_msg = f"❌ Arquivo muito grande ({file_size / 1024 / 1024:.1f} MB). Limite: {AppConfig.MAX_FILE_SIZE_MB} MB"
        logger.error(error_msg)
        return "", error_msg
    
    try:
        docx_loader = Docx2txtLoader(caminho)
        lista_documentos = docx_loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        
        if not documento or documento.strip() == '':
            raise ValueError("O arquivo Word parece estar vazio ou não foi possível extrair texto")
        
        logger.info(f"DOCX carregado: {caminho}")
        return documento, f"✅ Word carregado ({len(documento)} caracteres)"
        
    except Exception as e:
        error_msg = f"❌ Erro ao carregar arquivo Word: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


def carrega_csv(caminho: str, use_cache: bool = True) -> Tuple[str, str]:
    """
    Carrega o conteúdo de um arquivo CSV.
    
    Args:
        caminho: Caminho para o arquivo CSV
        use_cache: Se deve usar cache
        
    Returns:
        tuple: (conteúdo, mensagem de status)
    """
    if not os.path.exists(caminho):
        error_msg = f"❌ Arquivo não encontrado: {caminho}"
        logger.error(error_msg)
        return "", error_msg
    
    # Verificar tamanho do arquivo
    file_size = os.path.getsize(caminho)
    if file_size > AppConfig.MAX_FILE_SIZE_BYTES:
        error_msg = f"❌ Arquivo muito grande ({file_size / 1024 / 1024:.1f} MB). Limite: {AppConfig.MAX_FILE_SIZE_MB} MB"
        logger.error(error_msg)
        return "", error_msg
    
    try:
        csv_loader = CSVLoader(caminho, encoding='utf-8')
        lista_documentos = csv_loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        
        if not documento or documento.strip() == '':
            raise ValueError("O CSV parece estar vazio ou não foi possível extrair dados")
        
        # Contar linhas
        num_linhas = len(lista_documentos)
        
        logger.info(f"CSV carregado: {caminho} ({num_linhas} linhas)")
        return documento, f"✅ CSV carregado ({num_linhas} linhas, {len(documento)} caracteres)"
        
    except Exception as e:
        error_msg = f"❌ Erro ao carregar CSV: {str(e)}"
        logger.error(error_msg)
        return "", error_msg


def carrega_txt(caminho: str, use_cache: bool = True) -> Tuple[str, str]:
    """
    Carrega o conteúdo de um arquivo TXT.
    
    Args:
        caminho: Caminho para o arquivo TXT
        use_cache: Se deve usar cache
        
    Returns:
        tuple: (conteúdo, mensagem de status)
    """
    if not os.path.exists(caminho):
        error_msg = f"❌ Arquivo não encontrado: {caminho}"
        logger.error(error_msg)
        return "", error_msg
    
    # Verificar tamanho do arquivo
    file_size = os.path.getsize(caminho)
    if file_size > AppConfig.MAX_FILE_SIZE_BYTES:
        error_msg = f"❌ Arquivo muito grande ({file_size / 1024 / 1024:.1f} MB). Limite: {AppConfig.MAX_FILE_SIZE_MB} MB"
        logger.error(error_msg)
        return "", error_msg
    
    try:
        txt_loader = TextLoader(caminho, encoding='utf-8')
        lista_documentos = txt_loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        
        if not documento or documento.strip() == '':
            raise ValueError("O arquivo de texto parece estar vazio")
        
        logger.info(f"TXT carregado: {caminho}")
        return documento, f"✅ Texto carregado ({len(documento)} caracteres)"
        
    except Exception as e:
        error_msg = f"❌ Erro ao carregar TXT: {str(e)}"
        logger.error(error_msg)
        return "", error_msg
