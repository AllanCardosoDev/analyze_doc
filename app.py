"""
Analyse Doc - Aplicação principal
Sistema avançado de análise de documentos com IA
"""
import tempfile
import os
import logging
from typing import Generator, Optional
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage

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
    safe_session_state_set
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

# Aplicar estilos personalizados
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Inicializar configurações
config = AppConfig()
model_config = ModelConfig()


def inicializar_sessao():
    """Inicializa as variáveis de sessão necessárias."""
    defaults = {
        "memoria": ConversationBufferMemory(),
        "doc_memory_manager": None,
        "chain": None,
        "documento_carregado": False,
        "total_queries": 0,
        "k_chunks": config.DEFAULT_K_CHUNKS,
        "chunk_size": config.DEFAULT_CHUNK_SIZE,
        "chunk_overlap": config.DEFAULT_CHUNK_OVERLAP,
        "use_embeddings": False,
        "show_debug": False,
        "tokens_used": 0,
        "cost_accumulated": 0.0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def carrega_arquivos(tipo_arquivo: str, arquivo) -> tuple[str, str]:
    """
    Função unificada para carregar arquivos com tratamento de erros.
    
    Args:
        tipo_arquivo: Tipo do arquivo
        arquivo: Arquivo ou URL
        
    Returns:
        tuple: (conteúdo, mensagem de status)
    """
    if not arquivo:
        return "", "❌ Nenhum arquivo ou URL fornecido."
    
    try:
        if tipo_arquivo == "Site":
            return carrega_site(arquivo)
        
        elif tipo_arquivo == "Youtube":
            return carrega_youtube(arquivo)
        
        # Para outros tipos, criar arquivo temporário
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
            # Sempre remover arquivo temporário
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Erro ao limpar arquivo temporário: {e}")
    
    except Exception as e:
        logger.error(f"Erro ao carregar arquivo: {e}")
        return "", f"❌ Erro ao carregar arquivo: {str(e)}"


def carrega_modelo(provedor: str, modelo: str, api_key: str, tipo_arquivo: str, arquivo):
    """
    Carrega o modelo de IA e prepara o sistema com contexto completo do documento.
    """
    try:
        # Validar API key
        if not api_key:
            api_key = st.session_state.get(f'api_key_{provedor}', '')
        
        if not api_key:
            st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
            return
        
        # Validar formato da API key
        is_valid, message = validate_api_key(api_key, provedor)
        if not is_valid:
            st.error(f"⚠️ {message}")
            return
        
        # Mostrar progresso
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        
        # Carregar documento
        status_text.text("📄 Carregando documento...")
        progress_bar.progress(20)
        
        documento, status_msg = carrega_arquivos(tipo_arquivo, arquivo)
        
        if not documento or documento.startswith("❌"):
            st.error(status_msg if status_msg else "Documento não pôde ser carregado")
            progress_bar.empty()
            status_text.empty()
            return
        
        # Armazenar documento completo
        st.session_state['documento_completo'] = documento
        st.session_state['tamanho_documento'] = len(documento)
        st.session_state['tipo_arquivo'] = tipo_arquivo
        
        # Inicializar gerenciador de memória
        status_text.text("🔧 Processando documento...")
        progress_bar.progress(40)
        
        use_embeddings = st.session_state.get('use_embeddings', False)
        doc_manager = DocumentMemoryManager(use_embeddings=use_embeddings)
        st.session_state['doc_memory_manager'] = doc_manager
        
        # Processar documento
        chunk_size = st.session_state.get('chunk_size', config.DEFAULT_CHUNK_SIZE)
        chunk_overlap = st.session_state.get('chunk_overlap', config.DEFAULT_CHUNK_OVERLAP)
        
        processamento = doc_manager.process_document(
            documento, 
            tipo_arquivo,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        progress_bar.progress(60)
        
        # Preparar contexto do sistema baseado no tamanho do documento
        if len(documento) > config.SMALL_DOCUMENT_THRESHOLD:
            st.session_state['usando_documento_grande'] = True
            # Para documentos grandes, usar estratégia de recuperação
            documento_preview = doc_manager.get_document_preview(max_chars=2000)
            
            system_message = f"""Você é um assistente especializado em análise de documentos.

Você tem acesso a um documento {tipo_arquivo} com as seguintes informações:
- Total de páginas: {processamento['num_paginas']}
- Tamanho: {len(documento)} caracteres
- Processado em {processamento['total_chunks']} chunks

PREVIEW DO DOCUMENTO:
{documento_preview}

IMPORTANTE: Este é apenas um preview. Para cada pergunta do usuário, você receberá os trechos mais relevantes do documento completo.

INSTRUÇÕES:
1. Use SEMPRE as informações dos trechos fornecidos para responder
2. Se a informação não estiver nos trechos fornecidos, diga "Não encontrei essa informação específica nos trechos analisados"
3. Cite números de página quando disponíveis
4. Seja preciso e detalhado nas respostas
5. Se o usuário perguntar sobre capítulos, seções específicas, ou partes do documento, analise cuidadosamente os trechos fornecidos
6. Mantenha o contexto das perguntas anteriores quando relevante"""
        else:
            st.session_state['usando_documento_grande'] = False
            # Para documentos menores, incluir documento completo
            system_message = f"""Você é um assistente especializado em análise de documentos.

Você tem acesso completo ao seguinte documento {tipo_arquivo}:

====== DOCUMENTO COMPLETO ======
{documento}
====== FIM DO DOCUMENTO ======

Total de páginas: {processamento['num_paginas']}
Tamanho: {len(documento)} caracteres

INSTRUÇÕES:
1. Use as informações do documento para responder às perguntas
2. Seja preciso e detalhado
3. Cite números de página quando disponíveis
4. Se não encontrar a informação, seja honesto sobre isso
5. Mantenha o contexto das perguntas anteriores quando relevante"""
        
        # Criar template do prompt
        status_text.text("🤖 Configurando modelo de IA...")
        progress_bar.progress(80)
        
        template = ChatPromptTemplate.from_messages([
            ('system', system_message),
            ('placeholder', '{chat_history}'),
            ('user', '{input}')
        ])
        
        # Configurar modelo
        temperatura = model_config.PROVIDERS[provedor].get('temperatura_padrao', 0.7)
        
        if provedor == 'Groq':
            chat = ChatGroq(
                model=modelo,
                api_key=api_key,
                temperature=temperatura
            )
        else:  # OpenAI
            chat = ChatOpenAI(
                model=modelo,
                api_key=api_key,
                temperature=temperatura
            )
        
        chain = template | chat
        
        # Salvar na sessão
        st.session_state['chain'] = chain
        st.session_state['documento_carregado'] = True
        st.session_state['provedor_atual'] = provedor
        st.session_state['modelo_atual'] = modelo
        
        # Finalizar
        progress_bar.progress(100)
        status_text.text("✅ Pronto!")
        
        # Mostrar informações do documento
        st.sidebar.success(f"✅ Documento {tipo_arquivo} carregado com sucesso!")
        
        info_html = format_document_info({
            'tipo': tipo_arquivo,
            'tamanho': len(documento),
            'num_paginas': processamento['num_paginas'],
            'num_chunks': processamento['total_chunks']
        })
        st.sidebar.markdown(info_html, unsafe_allow_html=True)
        
        # Limpar progress
        import time
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}", exc_info=True)
        st.error(f"❌ Erro ao processar documento: {str(e)}")


def processar_pergunta_com_documento(
    input_usuario: str, 
    chain, 
    memoria: ConversationBufferMemory
) -> Generator[str, None, None]:
    """
    Processa perguntas usando chunks relevantes do documento de forma otimizada.
    """
    try:
        memory_manager = st.session_state.get('doc_memory_manager')
        if not memory_manager:
            yield "❌ Erro: Sistema não conseguiu acessar o documento. Por favor, tente recarregar."
            return
        
        # Obter configurações
        k_chunks = st.session_state.get('k_chunks', config.DEFAULT_K_CHUNKS)
        usando_doc_grande = st.session_state.get('usando_documento_grande', False)
        
        # Para documentos pequenos, não precisamos recuperar chunks
        if not usando_doc_grande:
            # Usar documento completo já no contexto do sistema
            pergunta_completa = input_usuario
        else:
            # Para documentos grandes, recuperar chunks relevantes
            chunks_relevantes = memory_manager.retrieve_relevant_chunks(
                input_usuario, 
                k=k_chunks
            )
            
            if not chunks_relevantes:
                yield "⚠️ Não consegui encontrar informações relevantes no documento para responder sua pergunta."
                return
            
            # Combinar chunks com metadados
            contexto_chunks = []
            for i, chunk in enumerate(chunks_relevantes):
                chunk_info = f"[Trecho {i+1}"
                if 'chunk_id' in chunk.metadata:
                    chunk_info += f" - ID: {chunk.metadata['chunk_id']}"
                chunk_info += "]\n" + chunk.page_content
                contexto_chunks.append(chunk_info)
            
            contexto_relevante = "\n\n---\n\n".join(contexto_chunks)
            
            # Criar prompt com contexto adicional
            prompt_adicional = f"""TRECHOS RELEVANTES DO DOCUMENTO PARA ESTA PERGUNTA:

{contexto_relevante}


Use APENAS as informações acima para responder à pergunta a seguir. Se a resposta não estiver nesses trechos, diga claramente que não encontrou a informação específica."""
            
            pergunta_completa = f"{prompt_adicional}\n\nPERGUNTA DO USUÁRIO: {input_usuario}"
        
        # Debug info
        if st.session_state.get('show_debug', False):
            with st.expander("🔍 Debug - Contexto Enviado"):
                st.text(f"Tamanho do contexto: {len(pergunta_completa)} caracteres")
                st.text(f"Tokens estimados: ~{estimate_tokens(pergunta_completa)}")
                if usando_doc_grande:
                    st.text(f"Chunks recuperados: {len(chunks_relevantes)}")
                st.code(pergunta_completa[:1000] + "..." if len(pergunta_completa) > 1000 else pergunta_completa)
        
        # Gerar resposta com streaming
        resposta_completa = ""
        for chunk in chain.stream({
            "input": pergunta_completa,
            "chat_history": memoria.buffer_as_messages
        }):
            if hasattr(chunk, 'content'):
                resposta_completa += chunk.content
            else:
                resposta_completa += str(chunk)
            yield resposta_completa
        
        # Atualizar estatísticas
        st.session_state['total_queries'] = st.session_state.get('total_queries', 0) + 1
        
        # Estimar tokens e custo
        input_tokens = estimate_tokens(pergunta_completa)
        output_tokens = estimate_tokens(resposta_completa)
        total_tokens = input_tokens + output_tokens
        
        st.session_state['tokens_used'] = st.session_state.get('tokens_used', 0) + total_tokens
        
        provedor = st.session_state.get('provedor_atual', 'Groq')
        modelo = st.session_state.get('modelo_atual', '')
        cost = estimate_cost(total_tokens, provedor, modelo)
        st.session_state['cost_accumulated'] = st.session_state.get('cost_accumulated', 0.0) + cost['total_estimated']
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {e}", exc_info=True)
        yield f"❌ Erro ao processar sua pergunta: {str(e)}"


def pagina_chat():
    """Interface principal do chat."""
    st.markdown('<h1 class="main-header">📑 Analyse Doc</h1>', unsafe_allow_html=True)
    
    chain = st.session_state.get('chain')
    
    if not chain or not st.session_state.get('documento_carregado', False):
        # Página de boas-vindas
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style='text-align: center; padding: 2rem;'>
                <h2>👋 Bem-vindo ao Analyse Doc!</h2>
                <p>Analise documentos com inteligência artificial de forma simples e poderosa.</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("📖 Como usar", expanded=True):
                st.markdown("""
                ### Passo a passo:
                
                1. **Selecione o tipo de documento** na barra lateral
                   - Suporta: PDF, Word, CSV, TXT, Sites e YouTube
                
                2. **Carregue seu documento**
                   - Faça upload do arquivo ou cole a URL
                
                3. **Escolha o modelo de IA**
                   - Groq (rápido e gratuito)
                   - OpenAI (mais avançado)
                
                4. **Configure sua API Key**
                   - Obtenha em: [Groq](https://console.groq.com) ou [OpenAI](https://platform.openai.com)
                
                5. **Clique em "Inicializar"**
                   - Aguarde o processamento
                
                6. **Comece a fazer perguntas!**
                   - Pergunte qualquer coisa sobre o documento
                """)
            
            with st.expander("✨ Recursos"):
                st.markdown("""
                - 🎯 **Análise Precisa**: Respostas baseadas no conteúdo real do documento
                - 📊 **Múltiplos Formatos**: Suporte para diversos tipos de arquivo
                - 💬 **Conversação Natural**: Mantenha contexto entre perguntas
                - 🔍 **Busca Inteligente**: Sistema avançado de recuperação de informações
                - 📈 **Estatísticas**: Acompanhe uso de tokens e custos
                """)
        
        st.stop()
    
    # Chat ativo
    memoria = st.session_state.get('memoria', ConversationBufferMemory())
    
    # Container para mensagens
    chat_container = st.container()
    
    with chat_container:
        # Exibir histórico
        for mensagem in memoria.buffer_as_messages:
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
    
    # Input do usuário
    input_usuario = st.chat_input("Faça perguntas sobre o documento carregado...")
    
    if input_usuario:
        # Exibir pergunta do usuário
        with chat_container:
            st.markdown(
                f'<div class="chat-message-human">👤 {input_usuario}</div>', 
                unsafe_allow_html=True
            )
        
        try:
            with st.spinner("🤔 Analisando..."):
                with chat_container:
                    resposta_container = st.empty()
                    
                    # Processar com streaming
                    for resposta_parcial in processar_pergunta_com_documento(
                        input_usuario, chain, memoria
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
            
        except Exception as e:
            with chat_container:
                st.error(f"❌ Erro ao processar resposta: {str(e)}")
            logger.error(f"Erro no chat: {e}", exc_info=True)


def sidebar():
    """Cria a barra lateral completa."""
    st.sidebar.header("🛠️ Configurações")
    
    tabs = st.sidebar.tabs(['📁 Upload', '🤖 Modelo', '⚙️ Avançado', '📊 Stats'])
    
    # === TAB: UPLOAD ===
    with tabs[0]:
        st.subheader("📁 Upload de Documentos")
        
        tipo_arquivo = st.selectbox(
            'Tipo de arquivo',
            FileTypes.SUPPORTED_TYPES,
            help="Selecione o tipo de documento que deseja analisar"
        )
        
        # Interface baseada no tipo
        if tipo_arquivo == 'Site':
            arquivo = st.text_input(
                'URL do site',
                placeholder="https://exemplo.com",
                help="Cole a URL completa do site que deseja analisar"
            )
        elif tipo_arquivo == 'Youtube':
            arquivo = st.text_input(
                'URL do vídeo',
                placeholder="https://www.youtube.com/watch?v=...",
                help="Cole a URL do vídeo do YouTube"
            )
        elif tipo_arquivo == 'Pdf':
            arquivo = st.file_uploader(
                'Upload do PDF',
                type=['pdf'],
                help=f"Tamanho máximo: {config.MAX_FILE_SIZE_MB} MB"
            )
        elif tipo_arquivo == 'Docx':
            arquivo = st.file_uploader(
                'Upload do Word',
                type=['docx'],
                help=f"Tamanho máximo: {config.MAX_FILE_SIZE_MB} MB"
            )
        elif tipo_arquivo == 'Csv':
            arquivo = st.file_uploader(
                'Upload do CSV',
                type=['csv'],
                help=f"Tamanho máximo: {config.MAX_FILE_SIZE_MB} MB"
            )
        elif tipo_arquivo == 'Txt':
            arquivo = st.file_uploader(
                'Upload do TXT',
                type=['txt'],
                help=f"Tamanho máximo: {config.MAX_FILE_SIZE_MB} MB"
            )
    
    # === TAB: MODELO ===
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
            help="Selecione o modelo específico"
        )
        
        # Campo de API Key
        api_key = st.text_input(
            f'API Key - {provedor}',
            type="password",
            value=st.session_state.get(f'api_key_{provedor}', ''),
            help=f"Sua chave de API do {provedor}"
        )
        st.session_state[f'api_key_{provedor}'] = api_key
        
        # Links úteis
        st.caption("🔑 Obtenha sua API Key:")
        if provedor == 'Groq':
            st.markdown("[console.groq.com](https://console.groq.com)")
        else:
            st.markdown("[platform.openai.com](https://platform.openai.com)")
    
    # === TAB: AVANÇADO ===
    with tabs[2]:
        st.subheader("⚙️ Configurações Avançadas")
        
        # Mostrar info do documento se carregado
        if 'doc_memory_manager' in st.session_state and 'documento_completo' in st.session_state:
            memory_manager = st.session_state['doc_memory_manager']
            info = memory_manager.get_document_info()
            
            st.markdown("**📄 Documento Atual:**")
            st.text(f"• Tipo: {info['tipo']}")
            st.text(f"• Tamanho: {info['tamanho']:,} chars")
            st.text(f"• Páginas: ~{info['num_paginas']}")
            st.text(f"• Chunks: {info['num_chunks']}")
            st.text(f"• Tokens: ~{info['estimated_tokens']:,}")
            
            st.markdown("---")
        
        # Configurações de processamento
        st.markdown("**🔧 Processamento:**")
        
        chunk_size = st.slider(
            "Tamanho dos chunks",
            min_value=config.MIN_CHUNK_SIZE,
            max_value=config.MAX_CHUNK_SIZE,
            value=st.session_state.get('chunk_size', config.DEFAULT_CHUNK_SIZE),
            step=500,
            help="Chunks maiores = mais contexto, mas mais tokens"
        )
        st.session_state['chunk_size'] = chunk_size
        
        k_chunks = st.slider(
            "Chunks por consulta",
            min_value=config.MIN_K_CHUNKS,
            max_value=config.MAX_K_CHUNKS,
            value=st.session_state.get('k_chunks', config.DEFAULT_K_CHUNKS),
            step=1,
            help="Mais chunks = mais contexto, mas mais tokens"
        )
        st.session_state['k_chunks'] = k_chunks
        
        # Opção de embeddings (requer recursos adicionais)
        use_embeddings = st.checkbox(
            "Usar busca vetorial (embeddings)",
            value=st.session_state.get('use_embeddings', False),
            help="Melhora a recuperação, mas usa mais recursos"
        )
        st.session_state['use_embeddings'] = use_embeddings
        
        # Debug mode
        show_debug = st.checkbox(
            "Modo debug",
            value=st.session_state.get('show_debug', False),
            help="Mostrar informações de debug durante o processamento"
        )
        st.session_state['show_debug'] = show_debug
    
    # === TAB: ESTATÍSTICAS ===
    with tabs[3]:
        st.subheader("📊 Estatísticas de Uso")
        
        total_queries = st.session_state.get('total_queries', 0)
        tokens_used = st.session_state.get('tokens_used', 0)
        cost = st.session_state.get('cost_accumulated', 0.0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Perguntas", total_queries)
            st.metric("Tokens", f"{tokens_used:,}")
        with col2:
            st.metric("Custo estimado", f"${cost:.4f}")
            if 'documento_carregado' in st.session_state:
                st.metric("Documento", "✅ Carregado")
        
        if st.button("Resetar estatísticas", use_container_width=True):
            st.session_state['total_queries'] = 0
            st.session_state['tokens_used'] = 0
            st.session_state['cost_accumulated'] = 0.0
            st.rerun()
    
    # === BOTÕES DE AÇÃO ===
    st.sidebar.markdown("---")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button('🚀 Inicializar', use_container_width=True, type="primary"):
            with st.spinner("⚙️ Processando..."):
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    with col2:
        if st.button('🗑️ Limpar Chat', use_container_width=True):
            st.session_state['memoria'] = ConversationBufferMemory()
            st.sidebar.success("✅ Chat limpo!")
            st.rerun()
    
    # Botão para novo documento
    if st.sidebar.button('📄 Novo Documento', use_container_width=True):
        # Limpar tudo relacionado ao documento
        keys_to_clear = [
            'chain', 'documento_completo', 'doc_memory_manager',
            'doc_chunks', 'vector_store', 'documento_carregado',
            'memoria', 'tamanho_documento', 'tipo_arquivo'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state['memoria'] = ConversationBufferMemory()
        st.sidebar.success("✅ Pronto para novo documento!")
        st.rerun()
    
    # === INFORMAÇÕES DO PROJETO ===
    st.sidebar.markdown("---")
    st.sidebar.caption("SOBRE")
    st.sidebar.info("""
    **Analyse Doc** v2.0
    
    Análise inteligente de documentos com IA
    
    Desenvolvido com Streamlit + LangChain
    """)


def main():
    """Função principal."""
    # Inicializar sessão
    inicializar_sessao()
    
    # Criar sidebar
    sidebar()
    
    # Página principal
    pagina_chat()


if __name__ == '__main__':
    main()
