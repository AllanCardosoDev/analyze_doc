import tempfile
import streamlit as st
import unicodedata
import re
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import *

TIPOS_ARQUIVOS_VALIDOS = [
    "Site", "Youtube", "Pdf", "Csv", "Txt"
]

CONFIG_MODELOS = {
    "Groq": {
        "modelos": [
            "distil-whisper-large-v3-en",
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-guard-3-8b",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "whisper-large-v3",
            "whisper-large-v3-turbo",
        ],
        "chat": ChatGroq,
    },
    "OpenAI": {
        "modelos": ["gpt-4o-mini", "gpt-4o", "o1-preview", "o1-mini"],
        "chat": ChatOpenAI,
    },
}

MEMORIA = ConversationBufferMemory()

def sanitize_text(text):
    """Sanitiza o texto para evitar problemas de codificação."""
    if not isinstance(text, str):
        return str(text)
    
    # Normalização Unicode
    text = unicodedata.normalize('NFKD', text)
    
    # Remove caracteres não-ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove caracteres de controle, mantendo quebras de linha e tabs
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    return text

def carrega_arquivos(tipo_arquivo, arquivo):
    """Função para carregar arquivos com tratamento de erros."""
    try:
        if tipo_arquivo == "Site":
            return carrega_site(arquivo)
        elif tipo_arquivo == "Youtube":
            return carrega_youtube(arquivo)
        elif tipo_arquivo == "Pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
                temp.write(arquivo.read())
                return carrega_pdf(temp.name)
        elif tipo_arquivo == "Csv":
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
                temp.write(arquivo.read())
                return carrega_csv(temp.name)
        elif tipo_arquivo == "Txt":
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
                temp.write(arquivo.read())
                return carrega_txt(temp.name)
    except Exception as e:
        return f"❌ Erro ao carregar arquivo: {e}"

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    """Carrega o modelo de IA e prepara o sistema para responder com base no documento."""
    if not api_key:
        st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
        return
    
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    if isinstance(documento, str) and (documento.startswith("❌") or documento.startswith("⚠️")):
        st.error(documento)
        return
    
    # Sanitizando o documento para evitar problemas de codificação
    documento_sanitizado = sanitize_text(documento)
    
    # Limitar o tamanho do documento para evitar problemas
    max_chars = 1500  # Ajuste conforme necessário
    documento_truncado = documento_sanitizado[:max_chars] if len(documento_sanitizado) > max_chars else documento_sanitizado
    
    system_message = f"""
    Você é um assistente chamado Analyse Doc.
    Aqui está o conteúdo do documento ({tipo_arquivo}) carregado:
    ###
    {documento_truncado}
    ###
    Responda com base nesse conteúdo.
    Se não conseguir acessar, informe ao usuário.
    
