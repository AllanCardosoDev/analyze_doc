import os
import time
import streamlit as st
from langchain_community.document_loaders import (
    WebBaseLoader,
    YoutubeLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader,
)
from fake_useragent import UserAgent

def carrega_site(url):
    """Carrega conteúdo de um site"""
    try:
        os.environ["USER_AGENT"] = UserAgent().random
        loader = WebBaseLoader(url, raise_for_status=True)
        documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in documentos]) or "Nenhum conteúdo extraído."
    except Exception as e:
        return f"Erro ao carregar site: {e}"

def carrega_youtube(video_url):
    """Carrega transcrição do YouTube"""
    try:
        loader = YoutubeLoader(video_url, add_video_info=False, language=["pt"])
        documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in documentos]) or "Nenhuma legenda disponível."
    except Exception as e:
        return f"Erro ao carregar vídeo: {e}"

def carrega_csv(caminho):
    """Carrega conteúdo de um arquivo CSV"""
    try:
        loader = CSVLoader(caminho)
        documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in documentos]) or "CSV vazio."
    except Exception as e:
        return f"Erro ao carregar CSV: {e}"

def carrega_pdf(caminho):
    """Carrega e extrai texto de um arquivo PDF."""
    try:
        loader = PyPDFLoader(caminho)
        documentos = loader.load()
        texto_extraido = "\n\n".join([doc.page_content for doc in documentos])

        if not texto_extraido.strip():
            return "⚠️ O PDF foi carregado, mas não contém texto extraível."

        return texto_extraido

    except Exception as e:
        return f"❌ Erro ao carregar PDF: {e}"

def carrega_txt(caminho):
    """Carrega conteúdo de um arquivo TXT"""
    try:
        loader = TextLoader(caminho)
        documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in documentos]) or "Arquivo TXT vazio."
    except Exception as e:
        return f"Erro ao carregar TXT: {e}"
