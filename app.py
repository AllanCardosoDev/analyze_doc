import tempfile
import streamlit as st
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

def carrega_arquivos(tipo_arquivo, arquivo):
    """Função para carregar arquivos com tratamento de erros."""
    try:
        if tipo_arquivo == "Site":
            return carrega_site(arquivo)
        elif tipo_arquivo == "Youtube":
            # Obtém o proxy configurado, se existir
            proxy = st.session_state.get("youtube_proxy", None)
            return carrega_youtube(arquivo, proxy=proxy)
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
    if documento.startswith("❌") or documento.startswith("⚠️"):
        st.error(documento)
        return
    
    system_message = f"""
    Você é um assistente chamado Analyse Doc.
    Aqui está o conteúdo do documento ({tipo_arquivo}) carregado:
    ###
    {documento[:2000]} # Limita para evitar sobrecarga de tokens
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
        st.stop()
    
    memoria = st.session_state.get("memoria", MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)
    
    input_usuario = st.chat_input("Fale com o Analyse Doc")
    if input_usuario:
        st.chat_message("human").markdown(input_usuario)
        resposta = st.chat_message("ai").write_stream(chain.stream({
            "input": input_usuario,
            "chat_history": memoria.buffer_as_messages
        }))
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta)
        st.session_state["memoria"] = memoria

def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    tabs = st.tabs(["Upload de Arquivos", "Seleção de Modelos", "Configurações"])
    
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
    
    with tabs[2]:
        st.subheader("Configurações do YouTube")
        proxy = st.text_input(
            "Proxy para YouTube (formato: http://usuario:senha@host:porta)",
            help="Use um proxy para contornar bloqueios do YouTube"
        )
        if proxy:
            st.session_state["youtube_proxy"] = proxy
        
        st.markdown("""
        **Dica para o YouTube:**
        Se você está enfrentando erros de "IP bloqueado", tente:
        1. Usar um proxy (configure acima)
        2. Usar uma VPN
        3. Esperar algumas horas e tentar novamente
        """)
    
    if st.button("Inicializar Analyse Doc", use_container_width=True):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    if st.button("Apagar Histórico de Conversa", use_container_width=True):
        st.session_state["memoria"] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
