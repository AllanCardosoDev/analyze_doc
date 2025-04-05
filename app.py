import tempfile
import os
import logging
import streamlit as st
from langchain.memory import ConversationBufferMemory

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from loaders import (
    carrega_site, 
    carrega_pdf, 
    carrega_csv, 
    carrega_txt,
    carrega_docx
)

from document_memory import DocumentMemoryManager

# Configurações de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurações da interface
st.set_page_config(
    page_title="Analyse Doc - Analise documentos com IA",
    page_icon="📑",
    layout="wide"
)

# Aplicar estilo padrão de chat, sem contraste excessivo
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #4F8BF9;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Estilos para chat padrão, sem tanto contraste */
    .chat-message-ai {
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
        background-color: rgba(100, 149, 237, 0.1);
        border-left: 2px solid #4F8BF9;
    }
    
    .chat-message-human {
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        border-radius: 0.5rem;
        background-color: rgba(220, 220, 220, 0.2);
        border-left: 2px solid #808080;
    }
    
    /* Botões com estilo mais suave */
    .stButton > button {
        background-color: #4F8BF9;
        color: white;
        font-weight: 500;
        border-radius: 0.3rem;
    }
    
    .stButton > button:hover {
        background-color: #3A66CC;
    }
</style>
""", unsafe_allow_html=True)

# Constantes
TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Pdf', 'Docx', 'Csv', 'Txt'
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

def carrega_arquivos(tipo_arquivo, arquivo):
    """Função para carregar arquivos com tratamento de erros."""
    if not arquivo:
        logger.warning("Nenhum arquivo ou URL fornecido.")
        return "❌ Nenhum arquivo ou URL fornecido."
        
    try:
        if tipo_arquivo == "Site":
            return carrega_site(arquivo)
        
        # Para outros tipos de arquivo, criar arquivo temporário
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{tipo_arquivo.lower()}") as temp:
            temp.write(arquivo.read())
            temp_path = temp.name
        
        try:
            if tipo_arquivo == "Pdf":
                return carrega_pdf(temp_path)
            elif tipo_arquivo == "Docx":
                return carrega_docx(temp_path)
            elif tipo_arquivo == "Csv":
                return carrega_csv(temp_path)
            elif tipo_arquivo == "Txt":
                return carrega_txt(temp_path)
        finally:
            # Sempre tentar remover o arquivo temporário
            try:
                os.unlink(temp_path)
            except Exception as cleanup_err:
                logger.error(f"Erro ao limpar arquivo temporário: {cleanup_err}")
    
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo: {e}")
        return f"❌ Erro ao carregar arquivo: {e}"

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    """Carrega o modelo de IA e prepara o sistema para responder com base no documento."""
    try:
        # Se não tiver API key, usa a da session_state
        if not api_key:
            api_key = st.session_state.get(f'api_key_{provedor}', '')
        
        if not api_key:
            st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
            return
        
        # Carregar documento com log detalhado
        documento = carrega_arquivos(tipo_arquivo, arquivo)
        
        if not documento or documento.startswith("❌"):
            st.error(documento if documento else "Documento não pôde ser carregado")
            return
        
        # Armazenar o documento completo na sessão
        st.session_state['documento_completo'] = documento
        st.session_state['tamanho_documento'] = len(documento)
        
        # Inicializar o gerenciador de memória de documentos
        if 'doc_memory_manager' not in st.session_state:
            st.session_state['doc_memory_manager'] = DocumentMemoryManager()
        
        # Para documentos grandes, processar usando o gerenciador de memória
        limite_tamanho = 50000  # Aumentamos o limite para 50K caracteres
        
        # Dependendo do tamanho do documento, usamos abordagens diferentes
        if len(documento) > limite_tamanho:
            # Para documentos muito grandes (mais de 50K caracteres)
            st.session_state['usando_documento_grande'] = True
            
            # Processar o documento com o gerenciador de memória
            memory_manager = st.session_state['doc_memory_manager']
            processamento = memory_manager.process_document(documento, tipo_arquivo)
            
            # Obter um preview do documento para o contexto inicial
            documento_preview = memory_manager.get_document_preview(max_chars=8000)
            
            # Informar o usuário sobre o uso do método para documentos grandes
            st.sidebar.info(f"📄 Documento grande ({len(documento)} caracteres) - Usando processamento avançado com {processamento['total_chunks']} chunks.")
            
            # Modificar a mensagem do sistema para enfatizar que o modelo tem acesso a todo o conteúdo
            system_message = f"""
            Você é um assistente chamado Analyse Doc especializado em analisar documentos.
            
            SOBRE O DOCUMENTO:
            - Tipo: {tipo_arquivo}
            - Tamanho: {len(documento)} caracteres
            - Dividido em: {processamento['total_chunks']} partes para processamento
            
            Este é um documento grande que foi processado usando técnicas avançadas. 
            Você tem acesso ao documento completo através de um sistema de recuperação
            de informações que fornecerá as partes relevantes do documento para cada pergunta.
            
            Aqui está um preview do conteúdo para você entender o contexto do documento:
            
            ####
            {documento_preview}
            ####
            
            Utilize as informações fornecidas para basear as suas respostas.
            Se a pergunta não puder ser respondida com as informações do documento, informe isso ao usuário.
            Seja detalhado e preciso em suas análises, sempre fundamentando suas respostas no conteúdo do documento.
            """
        else:
            # Para documentos de tamanho moderado, usamos o documento completo
            st.session_state['usando_documento_grande'] = False
            
            system_message = f"""
            Você é um assistente chamado Analyse Doc especializado em analisar documentos.
            Você possui acesso às seguintes informações vindas de um documento {tipo_arquivo}:
            
            ####
            {documento}
            ####
            
            Utilize as informações fornecidas para basear as suas respostas.
            Se a pergunta não puder ser respondida com as informações do documento, informe isso ao usuário.
            Seja detalhado e preciso em suas análises, sempre fundamentando suas respostas no conteúdo do documento.
            """
        
        template = ChatPromptTemplate.from_messages([
            ('system', system_message),
            ('placeholder', '{chat_history}'),
            ('user', '{input}')
        ])
        
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
        st.sidebar.success(f"✅ Documento {tipo_arquivo} carregado com sucesso! ({len(documento)} caracteres)")
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        st.error(f"❌ Erro ao processar documento: {e}")

def processar_pergunta_documento_grande(input_usuario, chain):
    """Processa perguntas para documentos grandes com recuperação de contexto."""
    try:
        # Obter o gerenciador de memória de documentos
        memory_manager = st.session_state.get('doc_memory_manager')
        if not memory_manager:
            return "Erro: Gerenciador de memória de documentos não inicializado."
        
        # Recuperar chunks relevantes para a pergunta
        chunks_relevantes = memory_manager.retrieve_relevant_chunks(input_usuario)
        
        # Combinar o conteúdo dos chunks relevantes
        contexto_relevante = "\n\n".join([chunk.page_content for chunk in chunks_relevantes])
        
        # Criar um prompt específico para esta pergunta
        prompt_especifico = f"""
        Com base nas seguintes seções do documento, responda à pergunta do usuário:
        
        SEÇÕES RELEVANTES DO DOCUMENTO:
        {contexto_relevante}
        
        PERGUNTA DO USUÁRIO:
        {input_usuario}
        
        Responda de forma detalhada e precisa, citando as informações relevantes do documento.
        Se a pergunta não puder ser respondida com as informações fornecidas, informe isso ao usuário.
        """
        
        # Usar o chain para gerar a resposta
        resposta = ""
        for chunk in chain.stream({"input": prompt_especifico, "chat_history": st.session_state['memoria'].buffer_as_messages}):
            if hasattr(chunk, 'content'):
                resposta += chunk.content
            else:
                resposta += str(chunk)
            yield resposta
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta para documento grande: {e}")
        yield f"Erro ao processar sua pergunta: {e}"

def pagina_chat():
    """Interface principal do chat - Simplificada, estilo chat padrão."""
    st.markdown('<h1 class="main-header">📑 Analyse Doc</h1>', unsafe_allow_html=True)
    
    chain = st.session_state.get('chain')
    if chain is None:
        st.info("Carregue um documento na barra lateral para começar a conversar.")
        
        with st.expander("ℹ️ Como usar o Analyse Doc"):
            st.markdown("""
            1. **Selecione o tipo de documento** na barra lateral.
            2. **Carregue o documento** (arquivo ou URL).
            3. **Escolha o modelo de IA** que deseja usar.
            4. **Adicione sua API Key** do provedor escolhido.
            5. **Inicialize o Analyse Doc** para começar a análise.
            6. **Faça perguntas** sobre o documento carregado.
            """)
            
        st.stop()
    
    # Recupera a memória da sessão
    memoria = st.session_state.get('memoria', ConversationBufferMemory())
    
    # Cria container para o chat (estilo mais padrão)
    chat_container = st.container()
    
    with chat_container:
        # Exibe o histórico de mensagens com estilo de chat padrão
        for mensagem in memoria.buffer_as_messages:
            if mensagem.type == 'ai':
                st.markdown(f'<div class="chat-message-ai">{mensagem.content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-message-human">{mensagem.content}</div>', unsafe_allow_html=True)
    
    # Campo de entrada do usuário
    input_usuario = st.chat_input("Faça perguntas sobre o documento carregado")
    
    if input_usuario:
        # Exibe a mensagem do usuário
        with chat_container:
            st.markdown(f'<div class="chat-message-human">{input_usuario}</div>', unsafe_allow_html=True)
        
        try:
            with st.spinner("Analisando..."):
                # Configuração para streaming de resposta
                with chat_container:
                    resposta_container = st.empty()
                    
                    # Verificar se estamos usando documento grande
                    if st.session_state.get('usando_documento_grande', False):
                        # Processar usando a abordagem para documentos grandes
                        for resposta_parcial in processar_pergunta_documento_grande(input_usuario, chain):
                            resposta_container.markdown(
                                f'<div class="chat-message-ai">{resposta_parcial}</div>',
                                unsafe_allow_html=True
                            )
                        resposta_completa = resposta_parcial
                    else:
                        # Abordagem padrão para documentos menores
                        resposta_parcial = []
                        for chunk in chain.stream({
                            "input": input_usuario,
                            "chat_history": memoria.buffer_as_messages
                        }):
                            # Adicionar o chunk à resposta parcial
                            if hasattr(chunk, 'content'):
                                resposta_parcial.append(chunk.content)
                            else:
                                resposta_parcial.append(str(chunk))
                            
                            # Atualizar a UI com a resposta parcial
                            resposta_container.markdown(
                                f'<div class="chat-message-ai">{"".join(resposta_parcial)}</div>',
                                unsafe_allow_html=True
                            )
                        
                        # Obter a resposta completa
                        resposta_completa = "".join(resposta_parcial)
            
            # Adiciona à memória
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta_completa)
            st.session_state['memoria'] = memoria
            
        except Exception as e:
            with chat_container:
                st.error(f"Erro ao processar resposta: {e}")

def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    st.sidebar.header("🛠️ Configurações")
    
    tabs = st.sidebar.tabs(['Upload de Arquivos', 'Seleção de Modelos', 'Processamento'])
    
    with tabs[0]:
        st.subheader("📁 Upload de Arquivos")
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo', TIPOS_ARQUIVOS_VALIDOS)
        
        # Interface de acordo com o tipo de arquivo
        if tipo_arquivo == 'Site':
            arquivo = st.text_input('Digite a URL do site', placeholder="https://exemplo.com")
        elif tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader('Faça o upload do arquivo PDF', type=['pdf'])
        elif tipo_arquivo == 'Docx':
            arquivo = st.file_uploader('Faça o upload do arquivo Word', type=['docx'])
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
    
    with tabs[2]:
        st.subheader("⚙️ Processamento")
        st.caption("Configurações para documentos grandes")
        
        # Tamanho máximo para considerar um documento "grande"
        max_tamanho_padrao = st.number_input(
            "Limite de tamanho para processamento padrão (caracteres)",
            min_value=5000,
            max_value=100000,
            value=50000,
            step=5000,
            help="Documentos maiores que este limite serão processados usando técnicas avançadas"
        )
        
        # Opção para sempre usar processamento avançado
        sempre_usar_processamento_avancado = st.checkbox(
            "Sempre usar processamento avançado",
            value=False,
            help="Ativar para usar processamento avançado mesmo para documentos pequenos"
        )
        
        # Guardar configurações na sessão
        st.session_state['max_tamanho_padrao'] = max_tamanho_padrao
        st.session_state['sempre_usar_processamento_avancado'] = sempre_usar_processamento_avancado
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button('Inicializar', use_container_width=True):
            with st.spinner("Carregando documento..."):
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    with col2:
        if st.button('Limpar Chat', use_container_width=True):
            st.session_state['memoria'] = ConversationBufferMemory()
            st.sidebar.success("✅ Histórico apagado")
    
    # Adicionar informações sobre o documento na sidebar
    if 'tipo_arquivo' in st.session_state and 'tamanho_documento' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.caption("DOCUMENTO ATUAL")
        st.sidebar.info(f"📄 {st.session_state['tipo_arquivo']} • {st.session_state['tamanho_documento']} caracteres")
        
        # Mostrar modo de processamento
        if st.session_state.get('usando_documento_grande', False):
            st.sidebar.success("🔄 Usando processamento avançado para documento grande")
        else:
            st.sidebar.info("🔄 Usando processamento padrão")
    
    # Informações do projeto (simplificado)
    st.sidebar.markdown("---")
    st.sidebar.caption("SOBRE")
    st.sidebar.info("Analyse Doc • Análise de documentos com IA")

def main():
    """Função principal."""
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == '__main__':
    main()
