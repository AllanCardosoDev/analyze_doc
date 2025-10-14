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
from typing import Optional, Generator

# Imports locais
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
    validate_api_key,
    format_document_info,
    estimate_tokens,
    estimate_cost,
    safe_session_state_get,
    safe_session_state_set,
    clear_session_state_prefix,
    setup_logging
)

# Configurar logging
setup_logging()
logger = logging.getLogger(__name__)

# Configurações da interface
st.set_page_config(
    page_title=AppConfig.APP_TITLE,
    page_icon=AppConfig.APP_ICON,
    layout=AppConfig.LAYOUT
)

# Aplicar estilos customizados
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def initialize_session_state():
    """Inicializa o estado da sessão com valores padrão."""
    defaults = {
        "memoria": ConversationBufferMemory(),
        "chain": None,
        "doc_memory_manager": None,
        "documento_completo": None,
        "tipo_arquivo": None,
        "tamanho_documento": 0,
        "num_paginas": 0,
        "usando_documento_grande": False,
        "doc_chunks": [],
        "doc_hash": "",
        "chunk_size": AppConfig.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": AppConfig.DEFAULT_CHUNK_OVERLAP,
        "k_chunks": AppConfig.DEFAULT_K_CHUNKS,
        "total_queries": 0,
        "use_embeddings": False,
        "messages_count": 0
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def carrega_arquivos(tipo_arquivo: str, arquivo) -> tuple:
    """
    Função para carregar arquivos com tratamento de erros.
    
    Args:
        tipo_arquivo: Tipo do arquivo a ser carregado
        arquivo: Arquivo ou URL
        
    Returns:
        tuple: (conteúdo, mensagem)
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
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{tipo_arquivo.lower()}") as temp:
            temp.write(arquivo.read())
            temp_path = temp.name
        
        try:
            if tipo_arquivo == "Pdf":
                resultado = carrega_pdf(temp_path)
            elif tipo_arquivo == "Docx":
                resultado = carrega_docx(temp_path)
            elif tipo_arquivo == "Csv":
                resultado = carrega_csv(temp_path)
            elif tipo_arquivo == "Txt":
                resultado = carrega_txt(temp_path)
            else:
                resultado = ("", f"❌ Tipo de arquivo não suportado: {tipo_arquivo}")
            
            return resultado
            
        finally:
            # Sempre tentar remover o arquivo temporário
            try:
                os.unlink(temp_path)
            except Exception as cleanup_err:
                logger.error(f"Erro ao limpar arquivo temporário: {cleanup_err}")
                
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo: {e}")
        return "", f"❌ Erro ao carregar arquivo: {str(e)}"


def test_api_key(provider: str, api_key: str, model: str) -> tuple:
    """
    Testa se a API key é válida fazendo uma chamada simples.
    
    Args:
        provider: Provedor (Groq ou OpenAI)
        api_key: Chave API
        model: Nome do modelo
        
    Returns:
        tuple: (sucesso: bool, mensagem: str)
    """
    try:
        if provider == "Groq":
            chat = ChatGroq(model=model, api_key=api_key, temperature=0)
        else:
            chat = ChatOpenAI(model=model, api_key=api_key, temperature=0)
        
        # Fazer uma chamada simples de teste
        response = chat.invoke("Hi")
        
        if response:
            return True, "✅ API key válida!"
        return False, "❌ Resposta inválida da API"
        
    except Exception as e:
        error_msg = str(e).lower()
        if "api key" in error_msg or "unauthorized" in error_msg or "invalid" in error_msg:
            return False, "❌ API key inválida ou sem permissão"
        elif "rate limit" in error_msg:
            return False, "⚠️ Limite de taxa excedido. Tente novamente mais tarde."
        else:
            return False, f"❌ Erro ao testar API: {str(e)[:100]}"


def carrega_modelo(provedor: str, modelo: str, api_key: str, tipo_arquivo: str, arquivo):
    """Carrega o modelo de IA e prepara o sistema para responder com base no documento."""
    
    # Validar API key
    if not api_key:
        api_key = safe_session_state_get(f'api_key_{provedor}', '')
    
    if not api_key:
        st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
        return
    
    # Validar formato da API key
    is_valid, msg = validate_api_key(api_key, provedor)
    if not is_valid:
        st.error(f"⚠️ {msg}")
        return
    
    # Mostrar progresso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Passo 1: Testar API key
        status_text.text("🔑 Validando API key...")
        progress_bar.progress(10)
        
        # Comentar teste para acelerar (opcional)
        # test_success, test_msg = test_api_key(provedor, api_key, modelo)
        # if not test_success:
        #     st.error(test_msg)
        #     return
        
        # Passo 2: Carregar documento
        status_text.text("📄 Carregando documento...")
        progress_bar.progress(30)
        
        documento, load_msg = carrega_arquivos(tipo_arquivo, arquivo)
        
        if not documento or documento.startswith("❌"):
            st.error(load_msg if load_msg else "Documento não pôde ser carregado")
            progress_bar.empty()
            status_text.empty()
            return
        
        # Mostrar mensagem de carregamento
        if load_msg.startswith("✅"):
            st.success(load_msg)
        
        # Passo 3: Processar documento
        status_text.text("⚙️ Processando documento...")
        progress_bar.progress(50)
        
        # Armazenar o documento completo na sessão
        safe_session_state_set('documento_completo', documento)
        safe_session_state_set('tamanho_documento', len(documento))
        safe_session_state_set('tipo_arquivo', tipo_arquivo)
        
        # Inicializar o gerenciador de memória de documentos
        use_embeddings = safe_session_state_get('use_embeddings', False)
        
        if 'doc_memory_manager' not in st.session_state or st.session_state['doc_memory_manager'] is None:
            st.session_state['doc_memory_manager'] = DocumentMemoryManager(use_embeddings=use_embeddings)
        
        memory_manager = st.session_state['doc_memory_manager']
        
        # Obter configurações de chunking
        chunk_size = safe_session_state_get('chunk_size', AppConfig.DEFAULT_CHUNK_SIZE)
        chunk_overlap = safe_session_state_get('chunk_overlap', AppConfig.DEFAULT_CHUNK_OVERLAP)
        
        # Processar com o gerenciador de memória
        processamento = memory_manager.process_document(
            documento, 
            tipo_arquivo,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Passo 4: Criar prompt e chain
        status_text.text("🤖 Configurando modelo de IA...")
        progress_bar.progress(70)
        
        # Para documentos grandes (mais de threshold caracteres)
        if len(documento) > AppConfig.SMALL_DOCUMENT_THRESHOLD:
            safe_session_state_set('usando_documento_grande', True)
            documento_preview = memory_manager.get_document_preview(max_chars=1500)
            
            system_message = f"""Você é um assistente especializado em análise de documentos.
            
Você possui acesso a informações de um documento do tipo {tipo_arquivo}.

Características do documento:
- Tamanho: {len(documento)} caracteres
- Páginas: aproximadamente {processamento['num_paginas']} páginas
- Processado em {processamento['total_chunks']} chunks

Preview do documento:
####
{documento_preview}
####

Este é apenas um trecho inicial. Você tem acesso ao documento completo através de um sistema 
de recuperação que fornecerá as informações mais relevantes para cada pergunta do usuário.

Instruções:
1. Utilize as informações do documento para responder às perguntas do usuário
2. Seja direto, preciso e útil nas suas respostas
3. Se não encontrar informação específica no contexto fornecido, indique isso claramente
4. Você pode fazer referência a perguntas anteriores quando relevante
5. Cite trechos específicos do documento quando apropriado
"""
        else:
            safe_session_state_set('usando_documento_grande', False)
            
            system_message = f"""Você é um assistente especializado em análise de documentos.

Você possui acesso completo a um documento do tipo {tipo_arquivo}:

####
{documento}
####

Instruções:
1. Utilize as informações do documento para responder às perguntas do usuário
2. Seja direto, preciso e útil nas suas respostas
3. Cite trechos específicos do documento quando apropriado
4. Você pode fazer referência a perguntas anteriores quando relevante
"""
        
        template = ChatPromptTemplate.from_messages([
            ('system', system_message),
            ('placeholder', '{chat_history}'),
            ('user', '{input}')
        ])
        
        # Passo 5: Inicializar modelo
        status_text.text("🚀 Inicializando modelo...")
        progress_bar.progress(90)
        
        config_modelo = ModelConfig.PROVIDERS[provedor]
        temperatura = config_modelo.get('temperatura_padrao', 0.7)
        
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
        safe_session_state_set('chain', chain)
        safe_session_state_set('current_provider', provedor)
        safe_session_state_set('current_model', modelo)
        
        # Finalizar
        progress_bar.progress(100)
        status_text.empty()
        progress_bar.empty()
        
        # Mostrar informações do documento
        info = memory_manager.get_document_info()
        st.sidebar.success(f"✅ Documento {tipo_arquivo} carregado com sucesso!")
        
        # Estatísticas
        st.sidebar.markdown(format_document_info(info), unsafe_allow_html=True)
        
        # Estimar custo
        if processamento.get('estimated_tokens'):
            cost_info = estimate_cost(
                processamento['estimated_tokens'],
                provedor,
                modelo
            )
            st.sidebar.info(
                f"💰 Custo estimado por consulta: ${cost_info['total_estimated']:.4f}"
            )
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}", exc_info=True)
        st.error(f"❌ Erro ao processar documento: {str(e)}")
        progress_bar.empty()
        status_text.empty()


def processar_pergunta_com_documento(
    input_usuario: str, 
    chain, 
    memoria: ConversationBufferMemory
) -> Generator[str, None, None]:
    """
    Processa perguntas usando chunks relevantes do documento.
    Utiliza a memória de conversação para manter contexto.
    """
    try:
        # Obter o gerenciador de memória de documentos
        memory_manager = safe_session_state_get('doc_memory_manager')
        
        if not memory_manager:
            yield "❌ Erro: Sistema não conseguiu acessar o documento. Por favor, tente recarregar."
            return
        
        # Obter o número de chunks configurado pelo usuário
        k_chunks = safe_session_state_get('k_chunks', AppConfig.DEFAULT_K_CHUNKS)
        
        # Recuperar chunks relevantes para a pergunta
        chunks_relevantes = memory_manager.retrieve_relevant_chunks(input_usuario, k=k_chunks)
        
        # Combinar o conteúdo dos chunks relevantes
        contexto_relevante = "\n\n".join([chunk.page_content for chunk in chunks_relevantes])
        
        # Criar um prompt que inclui os chunks relevantes
        prompt_adicional = f"""
Para responder à pergunta atual, use estas informações relevantes do documento:

{contexto_relevante}

Pergunta: {input_usuario}
"""
        
        # Usar o chain para gerar a resposta
        resposta = ""
        for chunk in chain.stream({
            "input": prompt_adicional,
            "chat_history": memoria.buffer_as_messages
        }):
            if hasattr(chunk, 'content'):
                resposta += chunk.content
            else:
                resposta += str(chunk)
            yield resposta
        
        # Incrementar contador de consultas
        total_queries = safe_session_state_get('total_queries', 0)
        safe_session_state_set('total_queries', total_queries + 1)
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {e}", exc_info=True)
        yield f"❌ Erro ao processar sua pergunta: {str(e)}"


def pagina_chat():
    """Interface principal do chat."""
    st.markdown('<h1 class="main-header">📑 Analyse Doc</h1>', unsafe_allow_html=True)
    
    chain = safe_session_state_get('chain')
    
    if chain is None:
        st.info("👈 Carregue um documento na barra lateral para começar a conversar.")
        
        with st.expander("ℹ️ Como usar o Analyse Doc"):
            st.markdown("""
            ### Guia Rápido de Uso
            
            **1. Selecione o tipo de documento** na barra lateral
            - Site, YouTube, PDF, Word, CSV ou TXT
            
            **2. Carregue o documento**
            - Para sites e YouTube: cole a URL
            - Para arquivos: faça o upload
            
            **3. Escolha o modelo de IA**
            - Groq: modelos rápidos e gratuitos
            - OpenAI: modelos mais avançados (GPT-4, etc)
            
            **4. Adicione sua API Key**
            - Groq: obtenha em https://console.groq.com
            - OpenAI: obtenha em https://platform.openai.com
            
            **5. Clique em "Inicializar"**
            - O sistema processará seu documento
            
            **6. Faça perguntas sobre o documento**
            - Digite suas perguntas no campo abaixo
            - O sistema buscará informações relevantes para responder
            
            ### Dicas
            - Para documentos grandes, o sistema usa chunks inteligentes
            - Você pode ajustar o tamanho dos chunks nas configurações
            - O histórico do chat é mantido durante a sessão
            - Use "Limpar Chat" para iniciar uma nova conversa
            """)
        
        # Mostrar estatísticas de uso se disponível
        if safe_session_state_get('total_queries', 0) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Consultas", safe_session_state_get('total_queries', 0))
            with col2:
                st.metric("💬 Mensagens", safe_session_state_get('messages_count', 0))
            with col3:
                if safe_session_state_get('tipo_arquivo'):
                    st.metric("📄 Tipo", safe_session_state_get('tipo_arquivo'))
        
        st.stop()
    
    # Recupera a memória da sessão
    memoria = safe_session_state_get('memoria', ConversationBufferMemory())
    
    # Container para o chat
    chat_container = st.container()
    
    with chat_container:
        # Exibir o histórico de mensagens
        messages = memoria.buffer_as_messages
        
        if len(messages) == 0:
            st.info("💡 Faça sua primeira pergunta sobre o documento!")
        
        for mensagem in messages:
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
        # Incrementar contador de mensagens
        messages_count = safe_session_state_get('messages_count', 0)
        safe_session_state_set('messages_count', messages_count + 1)
        
        # Exibir a mensagem do usuário
        with chat_container:
            st.markdown(
                f'<div class="chat-message-human">👤 {input_usuario}</div>', 
                unsafe_allow_html=True
            )
        
        try:
            with st.spinner("🔍 Analisando..."):
                # Configuração para streaming de resposta
                with chat_container:
                    resposta_container = st.empty()
                    
                    # Processar pergunta com documento
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
            safe_session_state_set('memoria', memoria)
            
        except Exception as e:
            logger.error(f"Erro ao processar resposta: {e}", exc_info=True)
            with chat_container:
                st.error(f"❌ Erro ao processar resposta: {str(e)}")


def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    st.sidebar.header("🛠️ Configurações")
    
    tabs = st.sidebar.tabs(['📁 Upload', '🤖 Modelo', '⚙️ Avançado'])
    
    # TAB 1: Upload de Arquivos
    with tabs[0]:
        st.subheader("Carregar Documento")
        
        tipo_arquivo = st.selectbox(
            'Tipo de documento',
            FileTypes.SUPPORTED_TYPES,
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
                'Upload PDF',
                type=['pdf'],
                help=f"Tamanho máximo: {AppConfig.MAX_FILE_SIZE_MB} MB"
            )
        elif tipo_arquivo == 'Docx':
            arquivo = st.file_uploader(
                'Upload Word',
                type=['docx'],
                help=f"Tamanho máximo: {AppConfig.MAX_FILE_SIZE_MB} MB"
            )
        elif tipo_arquivo == 'Csv':
            arquivo = st.file_uploader(
                'Upload CSV',
                type=['csv'],
                help=f"Tamanho máximo: {AppConfig.MAX_FILE_SIZE_MB} MB"
            )
        elif tipo_arquivo == 'Txt':
            arquivo = st.file_uploader(
                'Upload TXT',
                type=['txt'],
                help=f"Tamanho máximo: {AppConfig.MAX_FILE_SIZE_MB} MB"
            )
    
    # TAB 2: Seleção de Modelos
    with tabs[1]:
        st.subheader("Configurar IA")
        
        provedor = st.selectbox(
            'Provedor',
            ModelConfig.PROVIDERS.keys(),
            help="Escolha o provedor de IA"
        )
        
        modelo = st.selectbox(
            'Modelo',
            ModelConfig.PROVIDERS[provedor]['modelos'],
            help="Escolha o modelo específico"
        )
        
        # Campo de API key com persistência
        api_key_default = safe_session_state_get(f'api_key_{provedor}', '')
        api_key = st.text_input(
            f'API Key ({provedor})',
            type="password",
            value=api_key_default,
            help=f"Sua chave API do {provedor}"
        )
        
        if api_key:
            safe_session_state_set(f'api_key_{provedor}', api_key)
            
            # Validar formato
            is_valid, msg = validate_api_key(api_key, provedor)
            if is_valid:
                st.success("✅ Formato válido")
            else:
                st.warning(f"⚠️ {msg}")
    
    # TAB 3: Configurações Avançadas
    with tabs[2]:
        st.subheader("Processamento")
        
        # Mostrar informações do documento atual se disponível
        if 'doc_memory_manager' in st.session_state and st.session_state['doc_memory_manager'] is not None:
            memory_manager = st.session_state['doc_memory_manager']
            
            try:
                info = memory_manager.get_document_info()
                
                st.markdown("**📊 Documento Atual**")
                st.text(f"Tipo: {info.get('tipo', 'N/A')}")
                st.text(f"Tamanho: {info.get('tamanho', 0):,} caracteres")
                st.text(f"Páginas: ~{info.get('num_paginas', 0)}")
                st.text(f"Chunks: {info.get('num_chunks', 0)}")
                st.text(f"Tokens: ~{info.get('estimated_tokens', 0):,}")
                
                st.markdown("---")
            except Exception as e:
                logger.error(f"Erro ao obter info do documento: {e}")
        
        # Configurações de chunking
        st.caption("**Tamanho dos Chunks**")
        chunk_size = st.slider(
            "Caracteres por chunk",
            min_value=AppConfig.MIN_CHUNK_SIZE,
            max_value=AppConfig.MAX_CHUNK_SIZE,
            value=safe_session_state_get('chunk_size', AppConfig.DEFAULT_CHUNK_SIZE),
            step=500,
            help="Chunks menores = menos tokens, mas podem perder contexto"
        )
        safe_session_state_set('chunk_size', chunk_size)
        
        chunk_overlap = st.slider(
            "Sobreposição",
            min_value=0,
            max_value=500,
            value=safe_session_state_get('chunk_overlap', AppConfig.DEFAULT_CHUNK_OVERLAP),
            step=50,
            help="Overlap entre chunks para manter continuidade"
        )
        safe_session_state_set('chunk_overlap', chunk_overlap)
        
        st.caption("**Recuperação de Contexto**")
        k_chunks = st.slider(
            "Chunks por consulta",
            min_value=AppConfig.MIN_K_CHUNKS,
            max_value=AppConfig.MAX_K_CHUNKS,
            value=safe_session_state_get('k_chunks', AppConfig.DEFAULT_K_CHUNKS),
            step=1,
            help="Mais chunks = mais contexto, mas mais tokens"
        )
        safe_session_state_set('k_chunks', k_chunks)
        
        # Opção de usar embeddings (experimental)
        st.caption("**Recursos Experimentais**")
        use_embeddings = st.checkbox(
            "Usar busca vetorial",
            value=safe_session_state_get('use_embeddings', False),
            help="⚠️ Requer mais memória, mas melhora a precisão"
        )
        safe_session_state_set('use_embeddings', use_embeddings)
    
    # Botões de ação
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button('🚀 Inicializar', use_container_width=True, type="primary"):
            if not arquivo:
                st.sidebar.error("❌ Selecione um documento primeiro!")
            elif not api_key:
                st.sidebar.error("❌ Adicione sua API key!")
            else:
                with st.spinner("Processando..."):
                    carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    with col2:
        if st.button('🗑️ Limpar', use_container_width=True):
            # Limpar apenas o histórico do chat
            st.session_state['memoria'] = ConversationBufferMemory()
            st.session_state['messages_count'] = 0
            st.sidebar.success("✅ Chat limpo!")
            st.rerun()
    
    # Botão para reiniciar completamente
    if st.sidebar.button('🔄 Novo Documento', use_container_width=True):
        # Limpar tudo exceto API keys
        keys_to_keep = [k for k in st.session_state.keys() if k.startswith('api_key_')]
        keys_to_remove = [k for k in st.session_state.keys() if k not in keys_to_keep]
        
        for key in keys_to_remove:
            del st.session_state[key]
        
        initialize_session_state()
        st.sidebar.success("✅ Sistema reiniciado!")
        st.rerun()
    
    # Informações adicionais na sidebar
    st.sidebar.markdown("---")
    st.sidebar.caption("**📈 Estatísticas da Sessão**")
    
    if safe_session_state_get('chain') is not None:
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.metric("Consultas", safe_session_state_get('total_queries', 0))
        with col2:
            st.metric("Mensagens", safe_session_state_get('messages_count', 0))
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.caption("**ℹ️ Sobre**")
    st.sidebar.info(
        "**Analyse Doc**\n\n"
        "Análise inteligente de documentos com IA\n\n"
        "Desenvolvido por Allan Cardoso"
    )


def main():
    """Função principal."""
    # Inicializar estado da sessão
    initialize_session_state()
    
    # Renderizar sidebar
    sidebar()
    
    # Renderizar página de chat
    pagina_chat()


if __name__ == '__main__':
    main()
