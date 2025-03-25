import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from loaders import *

# Carregar variáveis de ambiente
load_dotenv()

# Configuração da página
st.set_page_config(page_title="Analyse Doc", layout="wide")

# Tipos de arquivos suportados
TIPOS_ARQUIVOS_VALIDOS = [
    "Site", "Youtube", "Pdf", "Docx", "Csv", "Txt"
]

# Configuração de modelos
CONFIG_MODELOS = {
    "Groq": {
        "modelos": [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-instant",
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
        ],
        "chat": ChatGroq,
    },
    "OpenAI": {
        "modelos": ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"],
        "chat": ChatOpenAI,
    },
}

# Inicialização do estado da sessão
if "memoria" not in st.session_state:
    st.session_state["memoria"] = ConversationBufferMemory()
if "chain" not in st.session_state:
    st.session_state["chain"] = None
if "youtube_proxy" not in st.session_state:
    st.session_state["youtube_proxy"] = None

def carrega_arquivos(tipo_arquivo, arquivo):
    """Função para carregar arquivos com tratamento de erros."""
    try:
        if tipo_arquivo == "Site":
            if not arquivo or not arquivo.strip():
                return "⚠️ URL não fornecida."
            return carrega_site(arquivo)
        
        elif tipo_arquivo == "Youtube":
            if not arquivo or not arquivo.strip():
                return "⚠️ URL do YouTube não fornecida."
            proxy = st.session_state.get("youtube_proxy", None)
            return carrega_youtube(arquivo, proxy=proxy)
        
        elif tipo_arquivo == "Pdf":
            if arquivo is None:
                return "⚠️ Nenhum arquivo PDF carregado."
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
                temp.write(arquivo.read())
                return carrega_pdf(temp.name)
        
        elif tipo_arquivo == "Docx":
            if arquivo is None:
                return "⚠️ Nenhum arquivo DOCX carregado."
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp:
                temp.write(arquivo.read())
                return carrega_docx(temp.name)
        
        elif tipo_arquivo == "Csv":
            if arquivo is None:
                return "⚠️ Nenhum arquivo CSV carregado."
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
                temp.write(arquivo.read())
                return carrega_csv(temp.name)
        
        elif tipo_arquivo == "Txt":
            if arquivo is None:
                return "⚠️ Nenhum arquivo TXT carregado."
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
                temp.write(arquivo.read())
                return carrega_txt(temp.name)
        
        else:
            return f"⚠️ Tipo de arquivo não suportado: {tipo_arquivo}"
    
    except Exception as e:
        return f"❌ Erro ao carregar arquivo: {e}"

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    """Carrega o modelo de IA e prepara o sistema para responder com base no documento."""
    if not api_key:
        st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
        return
    
    with st.spinner("Carregando documento..."):
        # Verifica se devemos usar o documento processado ou carregar um novo
        if "documento_processado" in st.session_state:
            documento = st.session_state.pop("documento_processado")
        else:
            documento = carrega_arquivos(tipo_arquivo, arquivo)
    
    # Verifica se ocorreu algum erro durante o carregamento
    if documento.startswith("❌") or documento.startswith("⚠️"):
        st.error(documento)
        return
    
    with st.spinner("Inicializando o modelo..."):
        # Monta o prompt do sistema
        system_message = f"""
        Você é um assistente chamado Analyse Doc.
        Aqui está o conteúdo do documento ({tipo_arquivo}) carregado:
        ###
        {documento[:2000]} # Limita para evitar sobrecarga de tokens
        ###
        
        Responda com base nesse conteúdo.
        Se não conseguir acessar alguma informação solicitada, informe ao usuário.
        """
        
        # Cria o template do chat
        template = ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("placeholder", "{chat_history}"),
            ("user", "{input}")
        ])
        
        # Configuração do chat
        try:
            chat = CONFIG_MODELOS[provedor]["chat"](model=modelo, api_key=api_key)
            chain = template | chat
            st.session_state["chain"] = chain
            st.success(f"✅ Analyse Doc inicializado com sucesso usando {provedor} - {modelo}")
        except Exception as e:
            st.error(f"❌ Erro ao inicializar o modelo: {e}")

def pagina_chat():
    """Cria a interface do chat e gerencia a conversa do usuário."""
    st.header("🤖 Analyse Doc - Assistente de Análise de Documentos", divider=True)
    
    # Verifica se o modelo foi carregado
    chain = st.session_state.get("chain")
    if chain is None:
        st.info("👈 Use a barra lateral para selecionar um documento e inicializar o Analyse Doc.")
        
        # Mostra um exemplo de uso
        with st.expander("Como usar o Analyse Doc"):
            st.markdown("""
            ### Instruções de uso:
            1. Na barra lateral, selecione o tipo de documento que deseja analisar
            2. Faça upload do arquivo ou informe a URL
            3. Escolha o provedor e modelo de IA
            4. Adicione sua chave de API
            5. Clique em "Inicializar Analyse Doc"
            6. Faça perguntas sobre o documento no chat abaixo
            
            ### Tipos de perguntas que você pode fazer:
            - "Qual é o tema principal deste documento?"
            - "Resuma as informações mais importantes"
            - "Quais são os pontos-chave mencionados?"
            - "Explique o conceito de X mencionado no texto"
            """)
        return
    
    # Exibe o histórico de mensagens
    memoria = st.session_state.get("memoria")
    for mensagem in memoria.buffer_as_messages:
        with st.chat_message(mensagem.type):
            st.markdown(mensagem.content)
    
    # Campo de entrada para nova mensagem
    input_usuario = st.chat_input("Faça uma pergunta sobre o documento...")
    if input_usuario:
        # Exibe a mensagem do usuário
        with st.chat_message("human"):
            st.markdown(input_usuario)
        
        # Processa e exibe a resposta
        with st.chat_message("ai"):
            with st.spinner("Analisando..."):
                try:
                    resposta_container = st.empty()
                    resposta_completa = ""
                    
                    # Stream da resposta
                    for chunk in chain.stream({
                        "input": input_usuario,
                        "chat_history": memoria.buffer_as_messages
                    }):
                        resposta_completa += chunk
                        resposta_container.markdown(resposta_completa + "▌")
                    
                    resposta_container.markdown(resposta_completa)
                except Exception as e:
                    st.error(f"❌ Erro ao gerar resposta: {e}")
                    resposta_completa = f"Desculpe, ocorreu um erro ao processar sua pergunta. Erro: {str(e)}"
                    resposta_container.markdown(resposta_completa)
        
        # Atualiza a memória com a nova interação
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta_completa)
        st.session_state["memoria"] = memoria

def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    # Título da barra lateral
    st.sidebar.title("Configurações")
    
    # Seção 1: Upload de Arquivos
    st.sidebar.header("Upload de Documentos")
    
    tipo_arquivo = st.sidebar.selectbox(
        "Tipo de documento", 
        TIPOS_ARQUIVOS_VALIDOS,
        help="Selecione o tipo de documento que deseja analisar"
    )
    
    # Interface específica por tipo de arquivo
    if tipo_arquivo in ["Site", "Youtube"]:
        arquivo = st.sidebar.text_input(
            f"URL do {tipo_arquivo.lower()}",
            help=f"Digite a URL do {tipo_arquivo.lower()} que deseja analisar"
        )
    else:
        # Mapeia tipos para extensões
        extensoes = {
            "Pdf": ["pdf"],
            "Docx": ["docx", "doc"],
            "Csv": ["csv"],
            "Txt": ["txt"]
        }
        
        arquivo = st.sidebar.file_uploader(
            f"Arquivo {tipo_arquivo.lower()}",
            type=extensoes.get(tipo_arquivo, [tipo_arquivo.lower()]),
            help=f"Faça upload do arquivo {tipo_arquivo.lower()} que deseja analisar"
        )
    
    # Seção 2: Modelo de IA
    st.sidebar.header("Modelo de IA")
    
    # Seleção de provedor
    provedor = st.sidebar.selectbox(
        "Provedor", 
        list(CONFIG_MODELOS.keys()),
        help="Selecione o provedor do modelo de IA"
    )
    
    # Seleção de modelo
    modelo = st.sidebar.selectbox(
        "Modelo", 
        CONFIG_MODELOS[provedor]["modelos"],
        help="Selecione o modelo específico"
    )
    
    # API Key
    api_key_env = os.getenv(f"{provedor.upper()}_API_KEY", "")
    api_key = st.sidebar.text_input(
        f"API Key ({provedor})", 
        value=api_key_env,
        type="password",
        help=f"Digite sua chave de API para {provedor}"
    )
    
    # Seção 3: Opções de Processamento
    st.sidebar.header("Opções de Processamento")
    
    # Opção de resumo
    gerar_resumo = st.sidebar.checkbox(
        "Gerar resumo automático",
        help="Cria um resumo do documento antes de processar"
    )
    
    # Comprimento do resumo (só exibe se gerar_resumo for True)
    if gerar_resumo:
        max_length = st.sidebar.slider(
            "Comprimento máximo do resumo",
            min_value=500,
            max_value=5000,
            value=1000,
            step=100,
            help="Número máximo de caracteres no resumo"
        )
        st.session_state["max_resumo_length"] = max_length
    
    # Seção 4: Configurações Adicionais
    with st.sidebar.expander("Configurações Adicionais"):
        # Proxy para YouTube
        youtube_proxy = st.text_input(
            "Proxy para YouTube",
            help="Formato: http://usuario:senha@host:porta"
        )
        if youtube_proxy:
            st.session_state["youtube_proxy"] = youtube_proxy
        
        # Dica para YouTube
        st.info("""
        **Dica para YouTube**: Se encontrar erro de "IP bloqueado", tente:
        1. Usar um proxy
        2. Usar uma VPN
        3. Esperar algumas horas
        """)
    
    # Botão de inicialização
    if st.sidebar.button("📄 Inicializar Analyse Doc", use_container_width=True, type="primary"):
        with st.spinner("Carregando..."):
            # Processa o documento se a opção de resumo estiver ativada
            if gerar_resumo:
                documento = carrega_arquivos(tipo_arquivo, arquivo)
                if not documento.startswith("❌") and not documento.startswith("⚠️"):
                    with st.spinner("Gerando resumo..."):
                        max_resumo_length = st.session_state.get("max_resumo_length", 1000)
                        documento = gera_resumo(documento, max_resumo_length)
                        st.session_state["documento_processado"] = documento
                        st.sidebar.success("✅ Resumo gerado com sucesso!")
            
            # Carrega o modelo
            carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    # Botão para limpar histórico
    if st.sidebar.button("🗑️ Limpar Histórico", use_container_width=True):
        st.session_state["memoria"] = ConversationBufferMemory()
        st.sidebar.success("✅ Histórico de conversa apagado!")
        st.rerun()

def main():
    """Função principal do aplicativo."""
    # Setup da barra lateral
    sidebar()
    
    # Conteúdo principal
    pagina_chat()

if __name__ == "__main__":
    main()
