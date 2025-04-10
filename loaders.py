import os
import logging
from time import sleep
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

# Configuração de logging
logger = logging.getLogger(__name__)

def carrega_site(url):
    """
    Carrega o conteúdo de um site web.
    Args:
        url (str): URL do site a ser carregado
    Returns:
        str: Conteúdo do site como texto
    """
    documento = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
            if not documento or documento.strip() == '':
                raise ValueError("Conteúdo do site está vazio")
            break
        except Exception as e:
            logger.warning(f"Tentativa {i+1}/5 falhou: {str(e)}")
            sleep(3)
    if documento == '':
        raise ValueError("Não foi possível carregar o site após múltiplas tentativas")
    return documento

def carrega_youtube(video_id):
    """
    Carrega a transcrição de um vídeo do Youtube.
    Args:
        video_id (str): ID ou URL completa do vídeo do Youtube
    Returns:
        str: Transcrição do vídeo como texto
    """
    try:
        # Se for URL completa, extrai apenas o ID do vídeo
        if "youtube.com" in video_id or "youtu.be" in video_id:
            if "youtube.com/watch?v=" in video_id:
                video_id = video_id.split("watch?v=")[1].split("&")[0]
            elif "youtu.be/" in video_id:
                video_id = video_id.split("youtu.be/")[1].split("?")[0]
        
        loader = YoutubeLoader(video_id, add_video_info=False, language=['pt'])
        lista_documentos = loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        
        if not documento or documento.strip() == '':
            raise ValueError("Não foi possível extrair a transcrição do vídeo")
        
        return documento
    except Exception as e:
        logger.error(f"Erro ao carregar vídeo do Youtube: {str(e)}")
        raise

def carrega_pdf(caminho):
    """
    Carrega o conteúdo de um arquivo PDF.
    Args:
        caminho (str): Caminho para o arquivo PDF
    Returns:
        str: Conteúdo do PDF como texto
    """
    try:
        loader = PyPDFLoader(caminho)
        lista_documentos = loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        if not documento or documento.strip() == '':
            raise ValueError("O PDF parece estar vazio ou não foi possível extrair texto")
        return documento
    except Exception as e:
        logger.error(f"Erro ao carregar PDF: {str(e)}")
        raise

def carrega_docx(caminho):
    """
    Carrega o conteúdo de um arquivo Word (DOCX).
    Args:
        caminho (str): Caminho para o arquivo DOCX
    Returns:
        str: Conteúdo do arquivo DOCX como texto
    """
    try:
        loader = Docx2txtLoader(caminho)
        lista_documentos = loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        if not documento or documento.strip() == '':
            raise ValueError("O arquivo Word parece estar vazio ou não foi possível extrair texto")
        return documento
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo Word: {str(e)}")
        raise

def carrega_csv(caminho):
    """
    Carrega o conteúdo de um arquivo CSV.
    Args:
        caminho (str): Caminho para o arquivo CSV
    Returns:
        str: Conteúdo do CSV como texto
    """
    try:
        loader = CSVLoader(caminho)
        lista_documentos = loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        if not documento or documento.strip() == '':
            raise ValueError("O CSV parece estar vazio ou não foi possível extrair dados")
        return documento
    except Exception as e:
        logger.error(f"Erro ao carregar CSV: {str(e)}")
        raise

def carrega_txt(caminho):
    """
    Carrega o conteúdo de um arquivo TXT.
    Args:
        caminho (str): Caminho para o arquivo TXT
    Returns:
        str: Conteúdo do TXT como texto
    """
    try:
        loader = TextLoader(caminho)
        lista_documentos = loader.load()
        documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
        if not documento or documento.strip() == '':
            raise ValueError("O arquivo de texto parece estar vazio")
        return documento
    except Exception as e:
        logger.error(f"Erro ao carregar TXT: {str(e)}")
        raise
