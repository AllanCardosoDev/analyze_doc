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
st.set_page_config(
    page_title="Analyse Doc", 
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
            "mixtral-8x7b-32768",
        ],
        "chat": ChatGroq,
    },
    "OpenAI": {
        "modelos": ["gpt-3.5-turbo", "gpt-4o-mini"],
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
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(arquivo.getvalue())
                    tmp_path = tmp.name
                
                result = carrega_pdf(tmp_path)
                
                # Limpar arquivo temporário
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
                return result
            except Exception as e:
                return f"❌ Erro ao processar PDF: {str(e)}"
        
        elif tipo_arquivo == "Docx":
            if arquivo is None:
                return "⚠️ Nenhum arquivo DOCX carregado."
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                    tmp.write(arquivo.getvalue())
                    tmp_path = tmp.name
                
                result = carrega_docx(tmp_path)
                
                # Limpar arquivo temporário
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
                return result
            except Exception as e:
                return f"❌ Erro ao processar DOCX: {str(e)}"
        
        elif tipo_arquivo == "Csv":
            if arquivo is None:
                return "⚠️ Nenhum arquivo CSV carregado."
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(arquivo.getvalue())
                    tmp_path = tmp.name
                
                result = carrega_csv(tmp_path)
                
                # Limpar arquivo temporário
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
                return result
            except Exception as e:
                return f"❌ Erro ao processar CSV: {str(e)}"
        
        elif tipo_arquivo == "Txt":
            if arquivo is None:
                return "⚠️ Nenhum arquivo TXT carregado."
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                    tmp.write(arquivo.getvalue())
                    tmp_path = tmp.name
                
                result = carrega_txt(tmp_path)
                
                # Limpar arquivo temporário
                try:
                    os.unlink(tmp_path)
                except:
                    pass
                    
                return result
            except Exception as e:
                return f"❌ Erro ao processar TXT: {str(e)}"
        
        else:
            return f"⚠️ Tipo de arquivo não suportado: {tipo_arquivo}"
    
    except Exception as e:
        return f"❌ Erro ao carregar arquivo: {str(e)}"

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
    if isinstance(documento, str) and (documento.startswith("❌") or documento.startswith("⚠️")):
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
            chat_class = CONFIG_MODELOS[provedor]["chat"]
            chat = chat_class(model=modelo, api_key=api_key)
            chain = template | chat
            st.session_state["chain"] = chain
            st.success(f"✅ Analyse Doc inicializado com sucesso usando {provedor} - {modelo}")
        except Exception as e:
            st.error(f"❌ Erro ao inicializar o modelo: {str(e)}")

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
    
    # Criamos um container dedicado para o histórico de mensagens
    chat_container = st.container()
    
    with chat_container:
        # Exibe as mensagens do histórico
        for mensagem in memoria.buffer_as_messages:
            with st.chat_message(mensagem.type):
                st.markdown(mensagem.content)
    
    # Campo de entrada para nova mensagem
    input_usuario = st.chat_input("Faça uma pergunta sobre o documento...")
    
    if input_usuario:
        # Exibe a mensagem do usuário
        with chat_container:
            with st.chat_message("human"):
                st.markdown(input_usuario)
            
            # Processa e exibe a resposta
            with st.chat_message("ai"):
                resposta_container = st.empty()
                resposta_completa = ""
                
                try:
                    # Tenta gerar a resposta
                    for chunk in chain.stream({
                        "input": input_usuario,
                        "chat_history": memoria.buffer_as_messages
                    }):
                        resposta_completa += chunk
                        resposta_container.markdown(resposta_completa + "▌")
                    
                    # Exibe a resposta final
                    resposta_container.markdown(resposta_completa)
                except Exception as e:
                    error_msg = f"Desculpe, ocorreu um erro ao processar sua pergunta: {str(e)}"
                    resposta_container.error(error_msg)
                    resposta_completa = error_msg
        
        # Atualiza a memória com a nova interação
        memoria.chat_memory.add_user_message(input_usuario)
        memoria.chat_memory.add_ai_message(resposta_completa)
        st.session_state["memoria"] = memoria

def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    st.sidebar.title("Configurações")
    
    # Seção 1: Upload de Arquivos
    st.sidebar.header("Upload de Documentos")
    
    tipo_arquivo = st.sidebar.selectbox(
        "Tipo de documento", 
        TIPOS_ARQUIVOS_VALIDOS
    )
    
    # Interface específica por tipo de arquivo
    if tipo_arquivo in ["Site", "Youtube"]:
        arquivo = st.sidebar.text_input(f"URL do {tipo_arquivo.lower()}")
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
            type=extensoes.get(tipo_arquivo, [tipo_arquivo.lower()])
        )
    
    # Seção 2: Modelo de IA
    st.sidebar.header("Modelo de IA")
    
    # Seleção de provedor
    provedor = st.sidebar.selectbox("Provedor", list(CONFIG_MODELOS.keys()))
    
    # Seleção de modelo
    modelo = st.sidebar.selectbox("Modelo", CONFIG_MODELOS[provedor]["modelos"])
    
    # API Key
    api_key_env = os.getenv(f"{provedor.upper()}_API_KEY", "")
    api_key = st.sidebar.text_input(
        f"API Key ({provedor})", 
        value=api_key_env,
        type="password"
    )
    
    # Seção 3: Opções de Processamento
    st.sidebar.header("Opções de Processamento")
    
    # Opção de resumo
    gerar_resumo = st.sidebar.checkbox("Gerar resumo automático")
    
    # Comprimento do resumo (só exibe se gerar_resumo for True)
    if gerar_resumo:
        max_length = st.sidebar.slider(
            "Comprimento máximo do resumo",
            min_value=500,
            max_value=5000,
            value=1000,
            step=100
        )
        st.session_state["max_resumo_length"] = max_length
    
    # Seção 4: Configurações Adicionais
    with st.sidebar.expander("Configurações Adicionais"):
        # Proxy para YouTube
        youtube_proxy = st.text_input("Proxy para YouTube (http://host:porta)")
        if youtube_proxy:
            st.session_state["youtube_proxy"] = youtube_proxy
        
        # Tema escuro/claro
        tema = st.radio("Tema", ["Claro", "Escuro"])
        if tema == "Escuro":
            st.markdown("""
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
            """, unsafe_allow_html=True)
    
    # Botão de inicialização
    if st.sidebar.button("📄 Inicializar Analyse Doc", type="primary", use_container_width=True):
        with st.spinner("Carregando..."):
            # Processa o documento se a opção de resumo estiver ativada
            if gerar_resumo:
                documento = carrega_arquivos(tipo_arquivo, arquivo)
                if not isinstance(documento, str) or (not documento.startswith("❌") and not documento.startswith("⚠️")):
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
    # Conteúdo principal e barra lateral
    pagina_chat()
    sidebar()

if __name__ == "__main__":
    main()
