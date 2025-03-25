import os
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import DocumentLoader
from utils import setup_logging, sanitize_input, encrypt_data
import logging

load_dotenv()
setup_logging()

TIPOS_ARQUIVOS_VALIDOS = ["Site", "Youtube", "Pdf", "Csv", "Txt"]
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

@st.cache_resource
def get_memory():
    return ConversationBufferMemory()

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    """Carrega o modelo de IA e prepara o sistema para responder com base no documento."""
    if not api_key:
        st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
        return

    try:
        documento = DocumentLoader.load(tipo_arquivo, arquivo)
    except Exception as e:
        st.error(f"❌ Erro ao carregar documento: {e}")
        logging.error(f"Erro ao carregar documento: {e}")
        return

    system_message = f"""
    Você é um assistente chamado Analyse Doc.
    Aqui está o conteúdo do documento ({tipo_arquivo}) carregado:
    ###
    {documento[:2000]}  # Limita para evitar sobrecarga de tokens
    ###
    Responda com base nesse conteúdo.
    Se não conseguir acessar, informe ao usuário.
    """
    
    template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])
    
    chat = CONFIG_MODELOS[provedor]["chat"](model=modelo, api_key=api_key)
    chain = template | chat
    st.session_state["chain"] = chain

def pagina_chat():
    """Cria a interface do chat e gerencia a conversa do usuário."""
    st.header("🤖 Bem-vindo ao Analyse Doc", divider=True)
    
    chain = st.session_state.get("chain")
    if chain is None:
        st.error("Carregue o Analyse Doc primeiro.")
        return

    memoria = get_memory()
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    input_usuario = st.chat_input("Fale com o Analyse Doc")
    if input_usuario:
        input_usuario = sanitize_input(input_usuario)
        st.chat_message("human").markdown(input_usuario)
        with st.spinner("Analyse Doc está pensando..."):
            resposta = chain.stream({
                "input": input_usuario,
                "chat_history": memoria.buffer_as_messages
            })
            resposta_completa = ""
            placeholder = st.chat_message("ai").empty()
            for chunk in resposta:
                resposta_completa += chunk
                placeholder.markdown(resposta_completa + "▌")
            placeholder.markdown(resposta_completa)
        
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta_completa)
        
        # Encrypta e salva o histórico de chat
        encrypted_history = encrypt_data(str(memoria.buffer_as_messages))
        st.session_state["encrypted_history"] = encrypted_history

def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    with st.sidebar:
        st.title("Configurações")
        
        tabs = st.tabs(["Upload de Arquivos", "Seleção de Modelos"])
        
        with tabs[0]:
            tipo_arquivo = st.selectbox("Selecione o tipo de arquivo", TIPOS_ARQUIVOS_VALIDOS)
            if tipo_arquivo in ["Site", "Youtube"]:
                arquivo = st.text_input(f"Digite a URL do {tipo_arquivo.lower()}")
            else:
                arquivo = st.file_uploader(f"Faça o upload do arquivo {tipo_arquivo.lower()}", type=[tipo_arquivo.lower()])
        
        with tabs[1]:
            provedor = st.selectbox("Selecione o provedor do modelo", list(CONFIG_MODELOS.keys()))
            modelo = st.selectbox("Selecione o modelo", CONFIG_MODELOS[provedor]["modelos"])
            api_key = st.text_input(f"Adicione a API key para {provedor}", type="password")
        
        if st.button("Inicializar Analyse Doc", use_container_width=True):
            with st.spinner("Inicializando Analyse Doc..."):
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
            st.success("Analyse Doc inicializado com sucesso!")
        
        if st.button("Apagar Histórico de Conversa", use_container_width=True):
            st.session_state["memoria"] = get_memory()
            st.success("Histórico de conversa apagado com sucesso!")

def main():
    st.set_page_config(page_title="Analyse Doc", page_icon="🤖", layout="wide")
    sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
