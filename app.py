import tempfile
import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import *
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()

# Configurar título e página
st.set_page_config(
    page_title="Analyse Doc - Analise documentos com IA",
    page_icon="🤖",
    layout="wide"
)

TIPOS_ARQUIVOS_VALIDOS = [
    "Site", "Pdf", "Docx", "Csv", "Txt"
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
    
    # Usa o documento processado se disponível, senão carrega normalmente
    if "documento_processado" in st.session_state:
        documento = st.session_state.pop("documento_processado")
    else:
        documento = carrega_arquivos(tipo_arquivo, arquivo)
    
    if not documento or isinstance(documento, str) and (documento.startswith("❌") or documento.startswith("⚠️")):
        st.error(documento if documento else "Documento não pôde ser carregado")
        return
    
    # Verificar idioma de saída
    idioma_codigo = st.session_state.get("idioma_codigo", "pt")
    
    # Se não for português e a tradução estiver disponível
    if idioma_codigo != "pt" and "tradutor_disponivel" in st.session_state and st.session_state["tradutor_disponivel"]:
        try:
            documento = traduz_texto(documento, idioma_codigo)
        except Exception as e:
            st.warning(f"Não foi possível traduzir o documento: {e}")
    
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
        st.success(f"✅ Modelo {modelo} carregado com sucesso! Documento pronto para análise.")
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")

def pagina_chat():
    """Cria a interface do chat e gerencia a conversa do usuário."""
    st.header("🤖 Bem-vindo ao Analyse Doc", divider=True)
    
    chain = st.session_state.get("chain")
    if chain is None:
        st.info("📚 Carregue um documento e inicialize o Analyse Doc para começar.")
        st.stop()
    
    memoria = st.session_state.get("memoria", ConversationBufferMemory())
    
    # Exibir histórico de mensagens
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)
    
    # Campo de entrada do usuário
    input_usuario = st.chat_input("Fale com o Analyse Doc sobre o documento carregado")
    
    if input_usuario:
        st.chat_message("human").markdown(input_usuario)
        
        try:
            with st.chat_message("ai"):
                resposta_stream = chain.stream({
                    "input": input_usuario,
                    "chat_history": memoria.buffer_as_messages
                })
                resposta = st.write_stream(resposta_stream)
            
            # Adicionar à memória
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta)
            st.session_state["memoria"] = memoria
            
        except Exception as e:
            st.error(f"❌ Erro ao processar resposta: {e}")

def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    st.sidebar.title("🛠️ Configurações")
    tabs = st.sidebar.tabs(["Upload de Arquivos", "Seleção de Modelos", "Processamento", "Configurações"])
    
    with tabs[0]:
        tipo_arquivo = st.selectbox("Selecione o tipo de arquivo", TIPOS_ARQUIVOS_VALIDOS)
        
        if tipo_arquivo in ["Site"]:
            arquivo = st.text_input(f"Digite a URL do {tipo_arquivo.lower()}")
        else:
            if tipo_arquivo == "Docx":
                arquivo = st.file_uploader(f"Faça o upload do arquivo {tipo_arquivo.lower()}", type=["docx"])
            elif tipo_arquivo == "Pdf":
                arquivo = st.file_uploader(f"Faça o upload do arquivo {tipo_arquivo.lower()}", type=["pdf"])
            elif tipo_arquivo == "Csv":
                arquivo = st.file_uploader(f"Faça o upload do arquivo {tipo_arquivo.lower()}", type=["csv"])
            elif tipo_arquivo == "Txt":
                arquivo = st.file_uploader(f"Faça o upload do arquivo {tipo_arquivo.lower()}", type=["txt"])
    
    with tabs[1]:
        provedor = st.selectbox("Selecione o provedor do modelo", list(CONFIG_MODELOS.keys()))
        modelo = st.selectbox("Selecione o modelo", CONFIG_MODELOS[provedor]["modelos"])
        
        # Obter API key das variáveis de ambiente, se disponível
        default_api_key = ""
        if provedor == "Groq":
            default_api_key = os.getenv("GROQ_API_KEY", "")
        elif provedor == "OpenAI":
            default_api_key = os.getenv("OPENAI_API_KEY", "")
            
        api_key = st.text_input(
            f"Adicione a API key para {provedor}",
            type="password",
            value=default_api_key
        )
    
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
        
        # Verificar se a tradução está disponível (exige um modelo de LLM)
        st.session_state["tradutor_disponivel"] = False  # Por padrão, não disponível
        
        st.checkbox("Extrair entidades", key="extrair_entidades", disabled=True,
                  help="Identifica nomes, organizações e outras entidades (em breve)")
        
        st.checkbox("Análise de sentimento", key="analise_sentimento", disabled=True,
                  help="Analisa o tom emocional do documento (em breve)")
    
    with tabs[3]:
        
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
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Inicializar Analyse Doc", use_container_width=True):
            with st.spinner("Carregando documento e inicializando..."):
                # Verificar se devemos processar o documento
                if st.session_state.get("gerar_resumo", False) and arquivo:
                    documento = carrega_arquivos(tipo_arquivo, arquivo)
                    if not documento.startswith("❌") and not documento.startswith("⚠️"):
                        st.info("Gerando resumo do documento...")
                        max_length = st.session_state.get("max_resumo_length", 1000)
                        documento = gera_resumo(documento, max_length)
                        st.session_state["documento_processado"] = documento
                
                # Inicia o modelo normalmente
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    with col2:
        if st.button("Apagar Histórico de Conversa", use_container_width=True):
            st.session_state["memoria"] = ConversationBufferMemory()
            st.success("✅ Histórico de conversa apagado!")

def main():
    """Função principal que configura a aplicação."""
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
