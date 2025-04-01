import tempfile
import os
from time import sleep
import streamlit as st
from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from langchain_community.document_loaders import (
    WebBaseLoader,
    YoutubeLoader, 
    CSVLoader, 
    PyPDFLoader, 
    TextLoader
)
from fake_useragent import UserAgent

# Configurações da interface
st.set_page_config(
    page_title="Analyse Doc - Analise documentos com IA",
    page_icon="📑",
    layout="wide"
)

# Aplicar estilo personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
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
</style>
""", unsafe_allow_html=True)

# Constantes
TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'Pdf', 'Csv', 'Txt'
]

CONFIG_MODELOS = {
    'Groq': {
        'modelos': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant', 'mixtral-8x7b-32768', 'gemma2-9b-it'],
        'chat': ChatGroq
    },
    'OpenAI': {
        'modelos': ['gpt-4o-mini', 'gpt-4o', 'o1-mini'],
        'chat': ChatOpenAI
    }
}

# Inicializar memória de conversa
if "memoria" not in st.session_state:
    st.session_state["memoria"] = ConversationBufferMemory()

# Funções para carregar documentos
def carrega_site(url):
    """Carrega texto de um site usando WebBaseLoader."""
    documento = ''
    for i in range(5):  # Tenta 5 vezes
        try:
            # Usar um user-agent aleatório para evitar bloqueios
            os.environ["USER_AGENT"] = UserAgent().random
            
            # Verificar formato da URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = "\n\n".join([doc.page_content for doc in lista_documentos])
            if documento:
                break
        except Exception as e:
            print(f"Tentativa {i+1} falhou: {e}")
            sleep(3)  # Aguarda 3 segundos antes de tentar novamente
            
    if not documento:
        return "⚠️ Não foi possível carregar o site após múltiplas tentativas."
        
    return documento

def carrega_youtube(video_id):
    """Carrega legendas de vídeos do YouTube."""
    try:
        # Extrai o video_id de uma URL completa, se for fornecida
        if "youtube.com" in video_id or "youtu.be" in video_id:
            if "youtube.com/watch?v=" in video_id:
                video_id = video_id.split("youtube.com/watch?v=")[1].split("&")[0]
            elif "youtu.be/" in video_id:
                video_id = video_id.split("youtu.be/")[1].split("?")[0]
        
        # Importa diretamente o youtube_transcript_api para maior compatibilidade
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.formatters import TextFormatter
        
        # Tenta primeiro com o idioma português, depois com inglês
        languages = ['pt', 'pt-BR', 'en']
        
        transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        
        # Formata as legendas em um texto contínuo
        formatter = TextFormatter()
        texto_formatado = formatter.format_transcript(transcripts)
        
        if not texto_formatado:
            # Fallback para método manual se o formatter falhar
            texto_completo = ""
            for entry in transcripts:
                texto_completo += f"{entry.get('text', '')} "
            return texto_completo
            
        return texto_formatado
        
    except Exception as e:
        mensagem_erro = str(e)
        if "IP" in mensagem_erro and "block" in mensagem_erro:
            return """❌ O YouTube está bloqueando as requisições do seu IP. Tente mais tarde."""
        return f"❌ Erro ao carregar YouTube: {e}"

def carrega_csv(caminho):
    """Carrega dados de arquivos CSV."""
    try:
        loader = CSVLoader(caminho)
        lista_documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        return f"❌ Erro ao carregar CSV: {e}"

def carrega_pdf(caminho):
    """Carrega e extrai texto de um PDF."""
    try:
        loader = PyPDFLoader(caminho)
        lista_documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        return f"❌ Erro ao carregar PDF: {e}"

def carrega_txt(caminho):
    """Carrega e extrai texto de um arquivo TXT."""
    try:
        loader = TextLoader(caminho)
        lista_documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        return f"❌ Erro ao carregar TXT: {e}"

def carrega_arquivos(tipo_arquivo, arquivo):
    """Função para carregar arquivos com tratamento de erros."""
    if not arquivo:
        return "❌ Nenhum arquivo ou URL fornecido."
        
    try:
        if tipo_arquivo == "Site":
            return carrega_site(arquivo)
        elif tipo_arquivo == "Youtube":
            return carrega_youtube(arquivo)
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
        return f"❌ Erro ao carregar arquivo: {e}"

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    """Carrega o modelo de IA e prepara o sistema para responder com base no documento."""
    # Se não tiver API key, usa a da session_state
    if not api_key:
        api_key = st.session_state.get(f'api_key_{provedor}', '')
    
    if not api_key:
        st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
        return
    
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    
    if not documento or isinstance(documento, str) and documento.startswith("❌"):
        st.error(documento if documento else "Documento não pôde ser carregado")
        return
    
    # Limitar o documento para casos de textos muito grandes
    max_chars = 8000
    documento_truncado = documento
    if len(documento) > max_chars:
        documento_truncado = documento[:max_chars] + f"\n\n[Documento truncado - exibindo {max_chars} de {len(documento)} caracteres]"
    
    system_message = f"""
    Você é um assistente chamado Analyse Doc especializado em analisar documentos.
    Você possui acesso às seguintes informações vindas de um documento {tipo_arquivo}:
    
    ####
    {documento_truncado}
    ####
    
    Utilize as informações fornecidas para basear as suas respostas.
    Se a pergunta não puder ser respondida com as informações do documento, informe isso ao usuário.
    """
    
    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])
    
    try:
        chat = CONFIG_MODELOS[provedor]['chat'](
            model=modelo, 
            api_key=api_key,
            temperature=0.7)
        chain = template | chat
        
        # Guarda na sessão
        st.session_state['chain'] = chain
        st.session_state['tipo_arquivo'] = tipo_arquivo
        st.session_state['tamanho_documento'] = len(documento)
        
        # Avisa o usuário que o documento foi carregado com sucesso
        st.success(f"✅ Documento {tipo_arquivo} carregado com sucesso! ({len(documento)} caracteres)")
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")

def pagina_chat():
    """Interface principal do chat."""
    st.markdown('<h1 class="main-header">📑 Analyse Doc</h1>', unsafe_allow_html=True)
    
    # Exibir informações do documento se disponível
    if 'tipo_arquivo' in st.session_state and 'tamanho_documento' in st.session_state:
        st.info(f"📄 **Documento:** {st.session_state['tipo_arquivo']} | **Tamanho:** {st.session_state['tamanho_documento']} caracteres")
    
    chain = st.session_state.get('chain')
    if chain is None:
        st.warning("⚠️ Carregue um documento e inicialize o Analyse Doc para começar.")
        st.stop()
    
    # Recupera a memória da sessão
    memoria = st.session_state.get('memoria', ConversationBufferMemory())
    
    # Exibe o histórico de mensagens com estilo personalizado
    for mensagem in memoria.buffer_as_messages:
        if mensagem.type == 'ai':
            st.markdown(f'<div class="chat-message-ai">{mensagem.content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message-human">{mensagem.content}</div>', unsafe_allow_html=True)
    
    # Campo de entrada do usuário
    input_usuario = st.chat_input("Fale com o Analyse Doc sobre o documento carregado")
    
    if input_usuario:
        # Exibe a mensagem do usuário
        st.markdown(f'<div class="chat-message-human">{input_usuario}</div>', unsafe_allow_html=True)
        
        try:
            with st.spinner("Analisando documento..."):
                # Usa invoke para evitar problemas com streaming
                resposta = chain.invoke({
                    "input": input_usuario,
                    "chat_history": memoria.buffer_as_messages
                })
                
                # Exibe a resposta
                st.markdown(f'<div class="chat-message-ai">{resposta.content}</div>', unsafe_allow_html=True)
            
            # Adiciona à memória
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta.content)
            st.session_state['memoria'] = memoria
            
        except Exception as e:
            st.error(f"❌ Erro ao processar resposta: {e}")

def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    st.sidebar.header("🛠️ Configurações")
    
    tabs = st.sidebar.tabs(['Upload de Arquivos', 'Seleção de Modelos'])
    
    with tabs[0]:
        st.subheader("📁 Upload de Arquivos")
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        
        # Interface de acordo com o tipo de arquivo
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a URL do site', placeholder="https://exemplo.com")
        elif tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a URL do vídeo', placeholder="https://youtube.com/watch?v=ID_VIDEO")
        elif tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo PDF', type=['pdf'])
        elif tipo_arquivo == 'Csv':
            arquivo = st.file_uploader('Faça o upload do arquivo CSV', type=['csv'])
        elif tipo_arquivo == 'Txt':
            arquivo = st.file_uploader('Faça o upload do arquivo TXT', type=['txt'])
    
    with tabs[1]:
        st.subheader("🤖 Seleção de Modelos")
        provedor = st.selectbox('Selecione o provedor do modelo', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo', CONFIG_MODELOS[provedor]['modelos'])
        
        # Salva a API key na sessão para reutilização
        api_key = st.text_input(
            f'API Key para {provedor}',
            type="password",
            value=st.session_state.get(f'api_key_{provedor}', ''))
        st.session_state[f'api_key_{provedor}'] = api_key
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button('Inicializar Analyse Doc', use_container_width=True):
            with st.spinner("Carregando documento..."):
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    with col2:
        if st.button('Apagar Histórico', use_container_width=True):
            st.session_state['memoria'] = ConversationBufferMemory()
            st.success("✅ Histórico de conversa apagado!")

def main():
    """Função principal."""
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()
