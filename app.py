import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import *

TIPOS_ARQUIVOS_VALIDOS = [
    "Site", "Youtube", "Pdf", "Docx", "Csv", "Txt"
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
        elif tipo_arquivo == "Docx":  # Novo tipo de arquivo
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp:
                temp.write(arquivo.read())
                return carrega_docx(temp.name)
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
    
    # Usa o documento processado se disponível, senão carrega normalmente
    if "documento_processado" in st.session_state:
        documento = st.session_state.pop("documento_processado")
    else:
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
    tabs = st.tabs(["Upload de Arquivos", "Seleção de Modelos", "Processamento", "Configurações"])
    
    with tabs[0]:
        tipo_arquivo = st.selectbox("Selecione o tipo de arquivo", TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo in ["Site", "Youtube"]:
            arquivo = st.text_input(f"Digite a URL do {tipo_arquivo.lower()}")
        else:
            if tipo_arquivo == "Docx":
                arquivo = st.file_uploader(f"Faça o upload do arquivo {tipo_arquivo.lower()}", type=["docx"])
            else:
                arquivo = st.file_uploader(f"Faça o upload do arquivo {tipo_arquivo.lower()}", type=[tipo_arquivo.lower()])
    
    with tabs[1]:
        provedor = st.selectbox("Selecione o provedor do modelo", list(CONFIG_MODELOS.keys()))
        modelo = st.selectbox("Selecione o modelo", CONFIG_MODELOS[provedor]["modelos"])
        api_key = st.text_input(f"Adicione a API key para {provedor}", type="password")
    
    with tabs[2]:
        st.subheader("Processamento avançado")
        
        st.checkbox("Gerar resumo automático", key="gerar_resumo", 
                  help="Cria um resumo do documento antes de processar")
        
        st.slider("Comprimento máximo do resumo", 500, 5000, 1000, key="max_resumo_length",
                help="Número máximo de caracteres no resumo")
        
        idiomas = {"Português": "pt", "Inglês": "en", "Espanhol": "es", "Francês": "fr"}
        idioma_selecionado = st.selectbox("Idioma de saída", list(idiomas.keys()), key="idioma_saida",
                               help="Traduzir o conteúdo para este idioma")
        st.session_state["idioma_codigo"] = idiomas[idioma_selecionado]
        
        st.checkbox("Extrair entidades", key="extrair_entidades", disabled=True,
                  help="Identifica nomes, organizações e outras entidades (em breve)")
        
        st.checkbox("Análise de sentimento", key="analise_sentimento", disabled=True,
                  help="Analisa o tom emocional do documento (em breve)")
    
    with tabs[3]:
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
        
        st.subheader("Preferências de interface")
        theme = st.selectbox("Tema", ["Claro", "Escuro"], key="theme")
        if theme == "Escuro":
            st.markdown(
                """
                <style>
                    .stApp {
                        background-color: #1E1E1E;
                        color: #FFFFFF;
                    }
                    .stTextInput, .stTextArea {
                        background-color: #2D2D2D;
                        color: #FFFFFF;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
    
    if st.button("Inicializar Analyse Doc", use_container_width=True):
        # Verificar se devemos processar o documento
        if st.session_state.get("gerar_resumo", False):
            documento = carrega_arquivos(tipo_arquivo, arquivo)
            if not documento.startswith("❌") and not documento.startswith("⚠️"):
                max_length = st.session_state.get("max_resumo_length", 1000)
                documento = gera_resumo(documento, max_length)
                st.session_state["documento_processado"] = documento
        
        # Inicia o modelo normalmente
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    if st.button("Apagar Histórico de Conversa", use_container_width=True):
        st.session_state["memoria"] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
