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
import logging

class DocumentLoader:
    @staticmethod
    def load(tipo_arquivo, arquivo):
        """Carrega o documento baseado no tipo de arquivo."""
        try:
            if tipo_arquivo == "Site":
                return DocumentLoader._carrega_site(arquivo)
            elif tipo_arquivo == "Youtube":
                return DocumentLoader._carrega_youtube(arquivo)
            elif tipo_arquivo == "Pdf":
                return DocumentLoader._carrega_pdf(arquivo)
            elif tipo_arquivo == "Csv":
                return DocumentLoader._carrega_csv(arquivo)
            elif tipo_arquivo == "Txt":
                return DocumentLoader._carrega_txt(arquivo)
            else:
                raise ValueError(f"Tipo de arquivo não suportado: {tipo_arquivo}")
        except Exception as e:
            logging.error(f"Erro ao carregar arquivo {tipo_arquivo}: {e}")
            raise

    @staticmethod
    def _carrega_site(url):
        """Carrega texto de um site usando WebBaseLoader."""
        for i in range(5):
            try:
                os.environ["USER_AGENT"] = UserAgent().random
                loader = WebBaseLoader(url, raise_for_status=True)
                lista_documentos = loader.load()
                return "\n\n".join([doc.page_content for doc in lista_documentos])
            except Exception as e:
                logging.warning(f"Tentativa {i+1} falhou ao carregar o site: {e}")
                sleep(3)
        raise Exception("Não foi possível carregar o site após 5 tentativas.")

    @staticmethod
    def _carrega_youtube(video_id):
        """Carrega legendas de vídeos do YouTube."""
        loader = YoutubeLoader(video_id, add_video_info=False, language=["pt"])
        lista_documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in lista_documentos])

    @staticmethod
    def _carrega_csv(arquivo):
        """Carrega dados de arquivos CSV."""
        with st.spinner("Carregando arquivo CSV..."):
            loader = CSVLoader(arquivo)
            lista_documentos = loader.load()
            return "\n\n".join([doc.page_content for doc in lista_documentos])

    @staticmethod
    def _carrega_pdf(arquivo):
        """Carrega e extrai texto de um PDF."""
        with st.spinner("Carregando arquivo PDF..."):
            loader = PyPDFLoader(arquivo)
            documentos = loader.load()
            texto = "\n\n".join([doc.page_content for doc in documentos])
            return texto if texto.strip() else "⚠️ O PDF não contém texto extraível."

    @staticmethod
    def _carrega_txt(arquivo):
        """Carrega e extrai texto de um arquivo TXT."""
        with st.spinner("Carregando arquivo TXT..."):
            loader = TextLoader(arquivo)
            lista_documentos = loader.load()
            return "\n\n".join([doc.page_content for doc in lista_documentos])
