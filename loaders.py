import os
from time import sleep
import streamlit as st
from langchain_community.document_loaders import (
    WebBaseLoader,
    YoutubeLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
from fake_useragent import UserAgent


def carrega_site(url):
    """Carrega texto de um site usando WebBaseLoader."""
    documento = ""
    for i in range(5):
        try:
            os.environ["USER_AGENT"] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = "\n\n".join([doc.page_content for doc in lista_documentos])
            break
        except Exception as e:
            print(f"Erro ao carregar o site {i+1}: {e}")
            sleep(3)

    if not documento:
        return "⚠️ Não foi possível carregar o site."

    return documento


def carrega_youtube(video_id):
    """Carrega legendas de vídeos do YouTube."""
    try:
        loader = YoutubeLoader(video_id, add_video_info=False, language=["pt"])
        lista_documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        return f"❌ Erro ao carregar YouTube: {e}"


def carrega_csv(caminho):
    """Carrega dados de arquivos CSV."""
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    return "\n\n".join([doc.page_content for doc in lista_documentos])


def carrega_pdf(caminho):
    """Carrega e extrai texto de um PDF."""
    try:
        loader = PyPDFLoader(caminho)
        documentos = loader.load()
        texto = "\n\n".join([doc.page_content for doc in documentos])
        return texto if texto.strip() else "⚠️ O PDF não contém texto extraível."
    except Exception as e:
        return f"❌ Erro ao carregar PDF: {e}"


def carrega_txt(caminho):
    """Carrega e extrai texto de um arquivo TXT."""
    loader = TextLoader(caminho)
    lista_documentos = loader.load()
    return "\n\n".join([doc.page_content for doc in lista_documentos])
