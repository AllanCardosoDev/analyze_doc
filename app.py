import tempfile
import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import carrega_site, carrega_youtube, carrega_pdf, carrega_docx, carrega_csv, carrega_txt
from dotenv import load_dotenv
import base64
from datetime import datetime
import traceback

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar título e página
st.set_page_config(
    page_title="Analyse Doc - Analise documentos com IA",
    page_icon="📑",
    layout="wide"
)

# Aplicar estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 2px solid #f0f2f6;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .document-info {
        background-color: #E3F2FD;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .pdf-section {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        border-left: 3px solid #1E88E5;
        margin-top: 10px;
    }
    
    .pdf-section h4 {
        color: #1E88E5;
        margin-bottom: 10px;
    }
    
    .chat-message-ai {
        background-color: #E3F2FD;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #1E88E5;
    }
    
    .chat-message-human {
        background-color: #F5F5F5;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #616161;
    }
    
    .sidebar-section {
        background-color: #F5F5F5;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Barra de carregamento personalizada */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

TIPOS_ARQUIVOS_VALIDOS = [
    "Site", "Youtube", "Pdf", "Docx", "Csv", "Txt"
]

CONFIG_MODELOS = {
    "Groq": {
        "modelos": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "chat": ChatGroq,
    },
    "OpenAI": {
        "modelos": ["gpt-4o-mini", "gpt-4o", "o1-mini"],
        "chat": ChatOpenAI,
    },
}

# Inicializar memória de conversa
if "memoria" not in st.session_state:
    st.session_state["memoria"] = ConversationBufferMemory()

def carrega_arquivos(tipo_arquivo, arquivo):
    """Função para carregar arquivos com tratamento de erros."""
    if not arquivo:
        return "❌ Nenhum arquivo ou URL fornecido."
        
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
                temp_path = temp.name
                
            # Ler o arquivo PDF
            resultado = carrega_pdf(temp_path)
            
            # Limpar o arquivo temporário
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return resultado
            
        elif tipo_arquivo == "Docx":
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp:
                temp.write(arquivo.read())
                temp_path = temp.name
                
            # Ler o arquivo DOCX
            resultado = carrega_docx(temp_path)
            
            # Limpar o arquivo temporário
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return resultado
            
        elif tipo_arquivo == "Csv":
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
                temp.write(arquivo.read())
                temp_path = temp.name
                
            # Ler o arquivo CSV
            resultado = carrega_csv(temp_path)
            
            # Limpar o arquivo temporário
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return resultado
            
        elif tipo_arquivo == "Txt":
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
                temp.write(arquivo.read())
                temp_path = temp.name
                
            # Ler o arquivo TXT
            resultado = carrega_txt(temp_path)
            
            # Limpar o arquivo temporário
            try:
                os.unlink(temp_path)
            except:
                pass
                
            return resultado
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        st.error(f"Stack trace: {stack_trace}")
        return f"❌ Erro ao carregar arquivo: {e}"

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    """Carrega o modelo de IA e prepara o sistema para responder com base no documento."""
    # Se não tiver API key, tenta pegar das variáveis de ambiente
    if not api_key:
        if provedor == "Groq":
            api_key = os.getenv("GROQ_API_KEY")
        elif provedor == "OpenAI":
            api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
        return
    
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    
    if not documento or isinstance(documento, str) and (documento.startswith("❌") or documento.startswith("⚠️")):
        st.error(documento if documento else "Documento não pôde ser carregado")
        return
    
    # Limitar o documento para evitar problemas com limites de token
    max_chars = 2000
    documento_truncado = documento[:max_chars]
    if len(documento) > max_chars:
        documento_truncado += f"\n\n[Documento truncado - exibindo {max_chars} de {len(documento)} caracteres]"
    
    system_message = f"""
    Você é um assistente chamado Analyse Doc.
    Aqui está o conteúdo do documento ({tipo_arquivo}) carregado:
    ###
    {documento_truncado}
    ###
    Responda com base nesse conteúdo.
    Se não conseguir acessar ou entender o conteúdo, informe ao usuário.
    """
    
    template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])
    
    try:
        chat = CONFIG_MODELOS[provedor]["chat"](model=modelo, api_key=api_key)
        chain = template | chat
        st.session_state["chain"] = chain
        st.session_state["documento_completo"] = documento
        st.session_state["tipo_arquivo"] = tipo_arquivo
        
        # Guardar metadados do documento para referência
        st.session_state["documento_meta"] = {
            "tipo": tipo_arquivo,
            "tamanho": len(documento),
            "data_processamento": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "modelo": modelo,
            "provedor": provedor
        }
        
        st.success(f"✅ Modelo {modelo} carregado com sucesso! Documento pronto para análise.")
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")

def pagina_chat():
    """Cria a interface do chat e gerencia a conversa do usuário."""
    st.markdown('<h1 class="main-header">📑 Analyse Doc</h1>', unsafe_allow_html=True)
    
    # Exibir informações do documento se disponível
    if "documento_meta" in st.session_state:
        meta = st.session_state["documento_meta"]
        with st.container():
            st.markdown('<div class="document-info">', unsafe_allow_html=True)
            st.markdown(f"**Tipo:** {meta['tipo']}")
            st.markdown(f"**Tamanho:** {meta['tamanho']} caracteres")
            st.markdown(f"**Modelo:** {meta['modelo']}")
            st.markdown(f"**Provedor:** {meta['provedor']}")
            st.markdown(f"**Processado em:** {meta['data_processamento']}")
            st.markdown('</div>', unsafe_allow_html=True)
    
    chain = st.session_state.get("chain")
    if chain is None:
        st.info("📚 Carregue um documento e inicialize o Analyse Doc para começar.")
        st.stop()
    
    memoria = st.session_state.get("memoria", ConversationBufferMemory())
    
    # Exibir histórico de mensagens com estilo melhorado
    for mensagem in memoria.buffer_as_messages:
        if mensagem.type == "ai":
            st.markdown(f'<div class="chat-message-ai">{mensagem.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-human">{mensagem.content}</div>', unsafe_allow_html=True)
    
    # Campo de entrada do usuário
    input_usuario = st.chat_input("Fale com o Analyse Doc sobre o documento carregado")
    
    if input_usuario:
        st.markdown(f'<div class="chat-message-human">{input_usuario}</div>', unsafe_allow_html=True)
        
        try:
            with st.spinner("Pensando..."):
                resposta_stream = chain.stream({
                    "input": input_usuario,
                    "chat_history": memoria.buffer_as_messages
                })
                
                placeholder = st.container()
                resposta_texto = ""
                
                # Exibição da resposta em stream com uma aparência melhor
                with placeholder:
                    message_placeholder = st.empty()
                    for chunk in resposta_stream:
                        resposta_texto += chunk
                        message_placeholder.markdown(
                            f'<div class="chat-message-ai">{resposta_texto}</div>', 
                            unsafe_allow_html=True
                        )
            
            # Adicionar à memória
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta_texto)
            st.session_state["memoria"] = memoria
            
        except Exception as e:
            st.error(f"❌ Erro ao processar resposta: {e}")

def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    st.sidebar.markdown('<h2 style="text-align: center; color: #1E88E5;">🛠️ Configurações</h2>', unsafe_allow_html=True)
    tabs = st.sidebar.tabs(["Upload de Arquivos", "Seleção de Modelos", "Configurações"])
    
    with tabs[0]:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📁 Upload de Arquivos")
        tipo_arquivo = st.selectbox(
            "Selecione o tipo de arquivo", 
            TIPOS_ARQUIVOS_VALIDOS,
            help="Escolha o tipo de arquivo que deseja analisar"
        )
        
        if tipo_arquivo in ["Site", "Youtube"]:
            arquivo = st.text_input(
                f"Digite a URL do {tipo_arquivo.lower()}", 
                placeholder=f"https://exemplo.com" if tipo_arquivo == "Site" else "https://youtube.com/watch?v=ID_VIDEO"
            )
        else:
            if tipo_arquivo == "Docx":
                arquivo = st.file_uploader(
                    f"Faça o upload do arquivo {tipo_arquivo.lower()}", 
                    type=["docx"],
                    help="Arquivos do Microsoft Word (.docx)"
                )
            elif tipo_arquivo == "Pdf":
                arquivo = st.file_uploader(
                    f"Faça o upload do arquivo {tipo_arquivo.lower()}", 
                    type=["pdf"],
                    help="Documentos PDF (.pdf)"
                )
            elif tipo_arquivo == "Csv":
                arquivo = st.file_uploader(
                    f"Faça o upload do arquivo {tipo_arquivo.lower()}", 
                    type=["csv"],
                    help="Planilhas em formato CSV (.csv)"
                )
            elif tipo_arquivo == "Txt":
                arquivo = st.file_uploader(
                    f"Faça o upload do arquivo {tipo_arquivo.lower()}", 
                    type=["txt"],
                    help="Arquivos de texto plano (.txt)"
                )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🤖 Seleção de Modelos")
        provedor = st.selectbox(
            "Selecione o provedor do modelo", 
            list(CONFIG_MODELOS.keys()),
            help="Escolha o provedor do modelo de IA"
        )
        modelo = st.selectbox(
            "Selecione o modelo", 
            CONFIG_MODELOS[provedor]["modelos"],
            help="Escolha o modelo específico para usar"
        )
        
        # Obter API key das variáveis de ambiente, se disponível
        default_api_key = ""
        if provedor == "Groq":
            default_api_key = os.getenv("GROQ_API_KEY", "")
        elif provedor == "OpenAI":
            default_api_key = os.getenv("OPENAI_API_KEY", "")
            
        api_key = st.text_input(
            f"Adicione a API key para {provedor}",
            type="password",
            value=default_api_key,
            help=f"Sua chave de API para {provedor}. Será salva apenas nesta sessão."
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🌐 Configurações do YouTube")
        proxy = st.text_input(
            "Proxy para YouTube (formato: http://usuario:senha@host:porta)",
            value=os.getenv("YOUTUBE_PROXY", ""),
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
        
        st.markdown("### 🎨 Preferências de interface")
        theme = st.selectbox(
            "Tema", 
            ["Claro", "Escuro"], 
            key="theme",
            help="Escolha o tema da interface"
        )
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
                    .document-info {
                        background-color: #2D2D2D;
                    }
                    .chat-message-ai {
                        background-color: #1E1E1E;
                        border-left: 3px solid #64B5F6;
                    }
                    .chat-message-human {
                        background-color: #2D2D2D;
                    }
                    .sidebar-section {
                        background-color: #2D2D2D;
                    }
                    .main-header {
                        color: #64B5F6;
                    }
                </style>
                """,
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Inicializar Analyse Doc", use_container_width=True):
            with st.spinner("Carregando documento e inicializando..."):
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    with col2:
        if st.button("Apagar Histórico", use_container_width=True):
            st.session_state["memoria"] = ConversationBufferMemory()
            st.success("✅ Histórico de conversa apagado!")

def main():
    """Função principal que configura a aplicação."""
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
