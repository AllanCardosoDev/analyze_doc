"""
Analyse Doc - Aplicação principal
Sistema avançado de análise de documentos com IA
"""
import tempfile
import os
import logging
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from typing import Generator, Optional, Tuple

from loaders import (
    carrega_site,
    carrega_youtube,
    carrega_pdf,
    carrega_csv,
    carrega_txt,
    carrega_docx
)
from document_memory import DocumentMemoryManager
from config import AppConfig, ModelConfig, FileTypes, CUSTOM_CSS
from utils import (
    setup_logging,
    validate_api_key,
    format_document_info,
    estimate_tokens,
    estimate_cost,
    safe_session_state_get,
    safe_session_state_set,
    format_file_size
)

# Configurar logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Configurações da interface
st.set_page_config(
    page_title=AppConfig.APP_TITLE,
    page_icon=AppConfig.APP_ICON,
    layout=AppConfig.LAYOUT,
    initial_sidebar_state="expanded"
)

# Aplicar estilos customizados
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Inicializar configurações
config = AppConfig()
model_config = ModelConfig()
file_types = FileTypes()


def init_session_state():
    """Inicializa todas as variáveis do session_state."""
    defaults = {
        "memoria": ConversationBufferMemory(),
        "chain": None,
        "doc_memory_manager": None,
        "documento_completo": None,
        "tipo_arquivo": None,
        "tamanho_documento": 0,
        "num_paginas": 0,
        "usando_documento_grande": False,
        "chunk_size": config.DEFAULT_CHUNK_SIZE,
        "k_chunks": config.DEFAULT_K_CHUNKS,
        "use_embeddings": False,
        "total_tokens_used": 0,
        "total_cost": 0.0,
        "message_count": 0,
        "document_loaded": False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def carrega_arquivos(tipo_arquivo: str, arquivo) -> Tuple[str, str]:
    """
    Função para carregar arquivos com tratamento de erros robusto.
    
    Args:
        tipo_arquivo: Tipo do arquivo (Site, Youtube, Pdf, etc.)
        arquivo: Arquivo ou URL
        
    Returns:
        tuple: (conteúdo do documento, mensagem de status)
    """
    if not arquivo:
        logger.warning("Nenhum arquivo ou URL fornecido.")
        return "", "❌ Nenhum arquivo ou URL fornecido."
    
    try:
        if tipo_arquivo == "Site":
            return carrega_site(arquivo)
        
        elif tipo_arquivo == "Youtube":
            return carrega_youtube(arquivo)
        
        # Para outros tipos de arquivo, criar arquivo temporário
        with tempfile.NamedTemporaryFile(
            delete=False, 
            suffix=f".{tipo_arquivo.lower()}"
        ) as temp:
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
            else:
                return "", f"❌ Tipo de arquivo não suportado: {tipo_arquivo}"
        
        finally:
            # Sempre tentar remover o arquivo temporário
            try:
                os.unlink(temp_path)
            except Exception as cleanup_err:
                logger.error(f"Erro ao limpar arquivo temporário: {cleanup_err}")
    
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo: {e}")
        return "", f"❌ Erro ao carregar arquivo: {e}"


def test_api_key(api_key: str, provider: str, model: str) -> Tuple[bool, str]:
    """
    Testa se a API key é válida fazendo uma chamada simples.
    
    Args:
        api_key: Chave API
        provider: Provedor (Groq, OpenAI)
        model: Nome do modelo
        
    Returns:
        tuple: (sucesso, mensagem)
    """
    try:
        if provider == 'Groq':
            chat = ChatGroq(model=model, api_key=api_key, max_tokens=10)
        else:
            chat = ChatOpenAI(model=model, api_key=api_key, max_tokens=10)
        
        # Fazer uma chamada de teste simples
        response = chat.invoke("Hi")
        return True, "✅ API key válida"
    
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
            return False, "❌ API key inválida ou não autorizada"
        elif "quota" in error_msg or "limit" in error_msg:
            return False, "❌ Limite de uso da API excedido"
        else:
            return False, f"❌ Erro ao validar API key: {str(e)[:100]}"


def carrega_modelo(
    provedor: str, 
    modelo: str, 
    api_key: str, 
    tipo_arquivo: str, 
    arquivo
) -> None:
    """
    Carrega o modelo de IA e prepara o sistema para responder.
    
    Args:
        provedor: Provedor do modelo (Groq, OpenAI)
        modelo: Nome do modelo
        api_key: Chave API
        tipo_arquivo: Tipo do arquivo
        arquivo: Arquivo ou URL
    """
    try:
        # Validar API key formato
        is_valid, message = validate_api_key(api_key, provedor)
        if not is_valid:
            st.error(message)
            return
        
        # Teste de conexão com a API
        with st.spinner("🔑 Validando API key..."):
            is_valid, message = test_api_key(api_key, provedor, modelo)
            if not is_valid:
                st.error(message)
                return
            st.success(message)
        
        # Carregar documento com barra de progresso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("📄 Carregando documento...")
        progress_bar.progress(20)
        
        documento, load_message = carrega_arquivos(tipo_arquivo, arquivo)
        
        if not documento or documento.startswith("❌"):
            st.error(load_message if load_message else "Documento não pôde ser carregado")
            progress_bar.empty()
            status_text.empty()
            return
        
        progress_bar.progress(40)
        status_text.text("✂️ Processando chunks...")
        
        # Armazenar o documento completo na sessão
        st.session_state['documento_completo'] = documento
        st.session_state['tamanho_documento'] = len(documento)
        st.session_state['tipo_arquivo'] = tipo_arquivo
        
        # Inicializar o gerenciador de memória de documentos
        use_embeddings = st.session_state.get('use_embeddings', False)
        
        if 'doc_memory_manager' not in st.session_state or use_embeddings:
            st.session_state['doc_memory_manager'] = DocumentMemoryManager(
                use_embeddings=use_embeddings
            )
        
        memory_manager = st.session_state['doc_memory_manager']
        
        # Processar documento com configurações customizadas
        chunk_size = st.session_state.get('chunk_size', config.DEFAULT_CHUNK_SIZE)
        k_chunks = st.session_state.get('k_chunks', config.DEFAULT_K_CHUNKS)
        
        progress_bar.progress(60)
        
        processamento = memory_manager.process_document(
            documento, 
            tipo_arquivo,
            chunk_size=chunk_size
        )
        
        progress_bar.progress(80)
        status_text.text("🤖 Inicializando modelo de IA...")
        
        # Preparar mensagem do sistema baseada no tamanho do documento
        if len(documento) > config.SMALL_DOCUMENT_THRESHOLD:
            st.session_state['usando_documento_grande'] = True
            documento_preview = memory_manager.get_document_preview(max_chars=1500)
            
            system_message = f"""Você é um assistente especializado em análise de documentos.

Você possui acesso a um documento {tipo_arquivo} com as seguintes características:
- Tamanho: {len(documento)} caracteres
- Páginas estimadas: {processamento['num_paginas']}
- Dividido em {processamento['total_chunks']} chunks para processamento

PREVIEW DO DOCUMENTO:
####
{documento_preview}
####

Este é apenas um trecho inicial. Você tem acesso ao documento completo através de um sistema 
de recuperação que fornecerá automaticamente as informações relevantes para cada pergunta.

INSTRUÇÕES:
- Utilize as informações do documento para responder às perguntas do usuário de forma precisa
- Seja direto, claro e útil nas suas respostas
- Cite trechos específicos quando relevante
- Se não encontrar informação específica, seja honesto sobre isso
- Você pode fazer referência a perguntas e respostas anteriores quando relevante
- Mantenha um tom profissional mas amigável
"""
        else:
            st.session_state['usando_documento_grande'] = False
            
            system_message = f"""Você é um assistente especializado em análise de documentos.

Você possui acesso ao seguinte documento {tipo_arquivo}:
####
{documento}
####

INSTRUÇÕES:
- Utilize as informações do documento para responder às perguntas do usuário de forma precisa
- Seja direto, claro e útil nas suas respostas
- Cite trechos específicos quando relevante
- Você pode fazer referência a perguntas e respostas anteriores quando relevante
- Mantenha um tom profissional mas amigável
"""
        
        # Criar template de prompt
        template = ChatPromptTemplate.from_messages([
            ('system', system_message),
            ('placeholder', '{chat_history}'),
            ('user', '{input}')
        ])
        
        # Inicializar o modelo
        temperatura = model_config.PROVIDERS[provedor].get('temperatura_padrao', 0.7)
        
        if provedor == 'Groq':
            chat = ChatGroq(
                model=modelo,
                api_key=api_key,
                temperature=temperatura
            )
        else:
            chat = ChatOpenAI(
                model=modelo,
                api_key=api_key,
                temperature=temperatura
            )
        
        chain = template | chat
        
        # Guardar na sessão
        st.session_state['chain'] = chain
        st.session_state['provedor_atual'] = provedor
        st.session_state['modelo_atual'] = modelo
        st.session_state['document_loaded'] = True
        
        # Salvar API key para reutilização
        st.session_state[f'api_key_{provedor}'] = api_key
        
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        # Exibir informações do documento
        info = memory_manager.get_document_info()
        st.sidebar.markdown(format_document_info(info), unsafe_allow_html=True)
        
        # Estatísticas
        tokens_estimados = estimate_tokens(documento)
        st.sidebar.success(f"✅ Documento processado com sucesso!")
        
        with st.sidebar.expander("📊 Estatísticas do Documento"):
            st.write(f"**Tokens estimados:** ~{tokens_estimados:,}")
            st.write(f"**Chunks criados:** {processamento['total_chunks']}")
            st.write(f"**Tamanho médio dos chunks:** {processamento['avg_chunk_size']} caracteres")
            if processamento['index_created']:
                st.write("**Índice vetorial:** ✅ Criado")
            else:
                st.write("**Busca:** Palavras-chave")
        
        logger.info(f"Modelo {provedor}/{modelo} carregado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        st.error(f"❌ Erro ao processar documento: {e}")
        
        # Limpar estados em caso de erro
        st.session_state['chain'] = None
        st.session_state['document_loaded'] = False


def processar_pergunta_com_documento(
    input_usuario: str, 
    chain, 
    memoria: ConversationBufferMemory
) -> Generator[str, None, None]:
    """
    Processa perguntas usando chunks relevantes do documento.
    
    Args:
        input_usuario: Pergunta do usuário
        chain: Chain do LangChain
        memoria: Memória de conversação
        
    Yields:
        str: Resposta em streaming
    """
    try:
        # Obter o gerenciador de memória de documentos
        memory_manager = st.session_state.get('doc_memory_manager')
        if not memory_manager:
            yield "❌ Erro: Sistema não conseguiu acessar o documento. Por favor, tente recarregar."
            return
        
        # Obter configurações
        k_chunks = st.session_state.get('k_chunks', config.DEFAULT_K_CHUNKS)
        
        # Recuperar chunks relevantes para a pergunta
        chunks_relevantes = memory_manager.retrieve_relevant_chunks(
            input_usuario, 
            k=k_chunks
        )
        
        if not chunks_relevantes:
            yield "❌ Não foi possível recuperar informações relevantes do documento."
            return
        
        # Combinar o conteúdo dos chunks relevantes
        contexto_relevante = "\n\n".join([
            f"[Trecho {i+1}]\n{chunk.page_content}" 
            for i, chunk in enumerate(chunks_relevantes)
        ])
        
        # Criar um prompt que inclui os chunks relevantes
        prompt_adicional = f"""
Com base nos seguintes trechos relevantes do documento, responda à pergunta:

{contexto_relevante}

Pergunta: {input_usuario}
"""
        
        # Usar o chain para gerar a resposta com streaming
        resposta_completa = ""
        
        for chunk in chain.stream({
            "input": prompt_adicional,
            "chat_history": memoria.buffer_as_messages
        }):
            if hasattr(chunk, 'content'):
                resposta_completa += chunk.content
            else:
                resposta_completa += str(chunk)
            
            yield resposta_completa
        
        # Atualizar estatísticas
        st.session_state['message_count'] = st.session_state.get('message_count', 0) + 1
        
        # Estimar tokens usados
        tokens_input = estimate_tokens(prompt_adicional)
        tokens_output = estimate_tokens(resposta_completa)
        total_tokens = tokens_input + tokens_output
        
        st.session_state['total_tokens_used'] = st.session_state.get('total_tokens_used', 0) + total_tokens
        
        # Estimar custo
        provedor = st.session_state.get('provedor_atual', 'Groq')
        modelo = st.session_state.get('modelo_atual', '')
        cost = estimate_cost(total_tokens, provedor, modelo)
        st.session_state['total_cost'] = st.session_state.get('total_cost', 0.0) + cost['total_estimated']
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {e}")
        yield f"❌ Erro ao processar sua pergunta: {e}"


def pagina_chat():
    """Interface principal do chat."""
    st.markdown('<h1 class="main-header">📑 Analyse Doc</h1>', unsafe_allow_html=True)
    
    chain = st.session_state.get('chain')
    
    if chain is None:
        st.info("👈 Carregue um documento na barra lateral para começar a conversar.")
        
        with st.expander("ℹ️ Como usar o Analyse Doc", expanded=True):
            st.markdown("""
            ### Passos para começar:
            
            1. **Selecione o tipo de documento** na barra lateral (Site, YouTube, PDF, etc.)
            2. **Carregue o documento** fazendo upload ou fornecendo a URL
            3. **Escolha o provedor e modelo de IA** que deseja usar
            4. **Adicione sua API Key** do provedor escolhido
            5. **Clique em "Inicializar"** para processar o documento
            6. **Faça perguntas** sobre o conteúdo do documento
            
            ### Recursos avançados:
            
            - **Documentos grandes**: Processamento automático com chunks
            - **Busca inteligente**: Sistema de recuperação de contexto relevante
            - **Memória de conversação**: O assistente lembra de perguntas anteriores
            - **Múltiplos formatos**: PDF, Word, CSV, TXT, Sites e vídeos do YouTube
            - **Embeddings opcionais**: Ative para busca semântica avançada
            
            ### Dicas:
            
            - Para documentos muito grandes, ajuste o tamanho dos chunks na aba "Processamento"
            - Ative embeddings para melhor qualidade de busca (requer mais recursos)
            - Use "Limpar Chat" para começar uma nova conversa mantendo o documento
            """)
        
        st.stop()
    
    # Recuperar memória da sessão
    memoria = st.session_state.get('memoria', ConversationBufferMemory())
    
    # Container para o chat
    chat_container = st.container()
    
    with chat_container:
        # Exibir o histórico de mensagens
        mensagens = memoria.buffer_as_messages
        
        if len(mensagens) == 0:
            st.markdown("""
            <div class='info-box'>
                💡 <strong>Dica:</strong> Faça perguntas específicas sobre o documento para obter 
                respostas mais precisas. Você pode pedir resumos, explicações, análises ou 
                buscar informações específicas.
            </div>
            """, unsafe_allow_html=True)
        
        for mensagem in mensagens:
            if mensagem.type == 'ai':
                st.markdown(
                    f'<div class="chat-message-ai">🤖 {mensagem.content}</div>', 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message-human">👤 {mensagem.content}</div>', 
                    unsafe_allow_html=True
                )
    
    # Campo de entrada do usuário
    input_usuario = st.chat_input("Faça perguntas sobre o documento carregado...")
    
    if input_usuario:
        # Exibir a mensagem do usuário
        with chat_container:
            st.markdown(
                f'<div class="chat-message-human">👤 {input_usuario}</div>', 
                unsafe_allow_html=True
            )
        
        try:
            with st.spinner("🤔 Analisando..."):
                # Container para a resposta
                with chat_container:
                    resposta_container = st.empty()
                    
                    # Processar com streaming
                    for resposta_parcial in processar_pergunta_com_documento(
                        input_usuario, 
                        chain, 
                        memoria
                    ):
                        resposta_container.markdown(
                            f'<div class="chat-message-ai">🤖 {resposta_parcial}</div>',
                            unsafe_allow_html=True
                        )
                        resposta_completa = resposta_parcial
            
            # Adicionar à memória
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta_completa)
            st.session_state['memoria'] = memoria
            
            # Atualizar interface
            st.rerun()
            
        except Exception as e:
            with chat_container:
                st.error(f"❌ Erro ao processar resposta: {e}")
            logger.error(f"Erro no chat: {e}")


def sidebar():
    """Cria a barra lateral para configurações."""
    st.sidebar.header("🛠️ Configurações")
    
    tabs = st.sidebar.tabs(['📁 Upload', '🤖 Modelos', '⚙️ Avançado'])
    
    # TAB 1: Upload de Arquivos
    with tabs[0]:
        st.subheader("📁 Carregar Documento")
        
        tipo_arquivo = st.selectbox(
            'Tipo de documento',
            file_types.SUPPORTED_TYPES,
            help="Selecione o tipo de documento que deseja analisar"
        )
        
        # Interface de acordo com o tipo de arquivo
        arquivo = None
        
        if tipo_arquivo == 'Site':
            arquivo = st.text_input(
                'URL do site',
                placeholder="https://exemplo.com",
                help="Cole a URL completa do site"
            )
        
        elif tipo_arquivo == 'Youtube':
            arquivo = st.text_input(
                'URL do vídeo',
                placeholder="https://www.youtube.com/watch?v=...",
                help="Cole a URL do vídeo do YouTube"
            )
        
        elif tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader(
                'Upload do arquivo PDF',
                type=['pdf'],
                help=f"Tamanho máximo: {config.MAX_FILE_SIZE_MB} MB"
            )
        
        elif tipo_arquivo == 'Docx':
            arquivo = st.file_uploader(
                'Upload do arquivo Word',
                type=['docx', 'doc'],
                help=f"Tamanho máximo: {config.MAX_FILE_SIZE_MB} MB"
            )
        
        elif tipo_arquivo == 'Csv':
            arquivo = st.file_uploader(
                'Upload do arquivo CSV',
                type=['csv'],
                help=f"Tamanho máximo: {config.MAX_FILE_SIZE_MB} MB"
            )
        
        elif tipo_arquivo == 'Txt':
            arquivo = st.file_uploader(
                'Upload do arquivo TXT',
                type=['txt'],
                help=f"Tamanho máximo: {config.MAX_FILE_SIZE_MB} MB"
            )
        
        # Avisos sobre o tipo de arquivo
        if tipo_arquivo in ['Site', 'Youtube']:
            st.caption("ℹ️ Certifique-se de que a URL está acessível publicamente")
        else:
            st.caption(f"ℹ️ Limite de tamanho: {config.MAX_FILE_SIZE_MB} MB")
    
    # TAB 2: Seleção de Modelos
    with tabs[1]:
        st.subheader("🤖 Modelo de IA")
        
        provedor = st.selectbox(
            'Provedor',
            list(model_config.PROVIDERS.keys()),
            help="Escolha o provedor de IA"
        )
        
        modelo = st.selectbox(
            'Modelo',
            model_config.PROVIDERS[provedor]['modelos'],
            help="Escolha o modelo específico"
        )
        
        # Campo de API key com persistência
        api_key_default = st.session_state.get(f'api_key_{provedor}', '')
        api_key = st.text_input(
            f'🔑 API Key ({provedor})',
            type="password",
            value=api_key_default,
            help="Sua chave API será armazenada apenas durante esta sessão"
        )
        
        if api_key and api_key != api_key_default:
            st.session_state[f'api_key_{provedor}'] = api_key
        
        # Informações sobre o modelo
        with st.expander("ℹ️ Sobre este modelo"):
            temp_padrao = model_config.PROVIDERS[provedor].get('temperatura_padrao', 0.7)
            st.write(f"**Temperatura padrão:** {temp_padrao}")
            st.write(f"**Max tokens:** {model_config.PROVIDERS[provedor].get('max_tokens', 4096)}")
            
            if provedor == 'OpenAI':
                st.write("**Observação:** Modelos da OpenAI têm custos por uso")
            else:
                st.write("**Observação:** Groq oferece tier gratuito generoso")
    
    # TAB 3: Configurações Avançadas
    with tabs[2]:
        st.subheader("⚙️ Processamento")
        
        # Exibir informações do documento atual
        if 'doc_memory_manager' in st.session_state and 'documento_completo' in st.session_state:
            memory_manager = st.session_state['doc_memory_manager']
            info = memory_manager.get_document_info()
            
            st.markdown("**📊 Documento Atual:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Tipo", info['tipo'])
                st.metric("Páginas", info['num_paginas'])
            with col2:
                st.metric("Chunks", info['num_chunks'])
                st.metric("Tokens", f"~{info['estimated_tokens']:,}")
            
            st.divider()
        
        # Configurações de chunking
        st.markdown("**✂️ Configurações de Chunks**")
        
        chunk_size = st.slider(
            "Tamanho dos chunks",
            min_value=config.MIN_CHUNK_SIZE,
            max_value=config.MAX_CHUNK_SIZE,
            value=st.session_state.get('chunk_size', config.DEFAULT_CHUNK_SIZE),
            step=500,
            help="Chunks maiores preservam contexto mas usam mais tokens"
        )
        st.session_state['chunk_size'] = chunk_size
        
        k_chunks = st.slider(
            "Chunks por consulta",
            min_value=config.MIN_K_CHUNKS,
            max_value=config.MAX_K_CHUNKS,
            value=st.session_state.get('k_chunks', config.DEFAULT_K_CHUNKS),
            step=1,
            help="Mais chunks = mais contexto, mas maior uso de tokens"
        )
        st.session_state['k_chunks'] = k_chunks
        
        st.divider()
        
        # Opção de embeddings
        st.markdown("**🧠 Busca Avançada**")
        use_embeddings = st.checkbox(
            "Usar embeddings vetoriais",
            value=st.session_state.get('use_embeddings', False),
            help="Melhora a qualidade da busca mas requer mais recursos"
        )
        st.session_state['use_embeddings'] = use_embeddings
        
        if use_embeddings:
            st.info("ℹ️ Embeddings serão baixados no primeiro uso (~400MB)")
    
    # Botões de ação
    st.sidebar.divider()
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button('🚀 Inicializar', use_container_width=True, type="primary"):
            if not arquivo:
                st.sidebar.error("❌ Selecione ou forneça um documento")
            elif not api_key:
                st.sidebar.error("❌ Forneça uma API Key")
            else:
                with st.spinner("Processando..."):
                    carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    with col2:
        if st.button('🗑️ Limpar Chat', use_container_width=True):
            st.session_state['memoria'] = ConversationBufferMemory()
            st.session_state['message_count'] = 0
            st.sidebar.success("✅ Chat limpo")
            st.rerun()
    
    # Botão para novo documento
    if st.sidebar.button('📄 Novo Documento', use_container_width=True):
        # Limpar todos os estados relacionados ao documento
        keys_to_clear = [
            'chain', 'documento_completo', 'tipo_arquivo', 'tamanho_documento',
            'num_paginas', 'doc_chunks', 'doc_hash', 'vector_store',
            'usando_documento_grande', 'memoria', 'document_loaded'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state['memoria'] = ConversationBufferMemory()
        st.sidebar.success("✅ Pronto para novo documento")
        st.rerun()
    
    # Estatísticas de uso
    if st.session_state.get('document_loaded', False):
        st.sidebar.divider()
        st.sidebar.caption("📈 ESTATÍSTICAS DA SESSÃO")
        
        with st.sidebar.expander("Ver estatísticas"):
            message_count = st.session_state.get('message_count', 0)
            total_tokens = st.session_state.get('total_tokens_used', 0)
            total_cost = st.session_state.get('total_cost', 0.0)
            
            st.write(f"**Mensagens:** {message_count}")
            st.write(f"**Tokens usados:** ~{total_tokens:,}")
            st.write(f"**Custo estimado:** ${total_cost:.4f}")
    
    # Informações do projeto
    st.sidebar.divider()
    st.sidebar.caption("SOBRE")
    st.sidebar.info(
        "**Analyse Doc** v2.0\n\n"
        "Sistema avançado de análise de documentos com IA\n\n"
        "Desenvolvido com Streamlit, LangChain e modelos de linguagem de última geração."
    )


def main():
    """Função principal."""
    # Inicializar session state
    init_session_state()
    
    # Renderizar sidebar
    with st.sidebar:
        sidebar()
    
    # Renderizar página principal
    pagina_chat()


if __name__ == '__main__':
    main()
