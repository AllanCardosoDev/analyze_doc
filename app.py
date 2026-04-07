"""
Analyse Doc - Aplicação principal
Sistema avançado de análise de documentos com IA
Versão 2.0 - Com SmartRetriever e Diagnóstico
"""
import tempfile
import os
import logging
from typing import Generator, Optional
import streamlit as st
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

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
from diagnostico import adicionar_interface_diagnostico, DocumentDiagnostic
from melhorias_recuperacao import SmartRetriever, integrar_smart_retriever

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
        "memoria": ChatMessageHistory(),
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
        "cost_accumulated": 0.0,
        "smart_retriever": None,
        "estrutura_documento": None
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
            else:
                return "", f"❌ Tipo de arquivo não suportado: {tipo_arquivo}"
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
    Versão melhorada com SmartRetriever integrado.
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
        
        # === NOVA INTEGRAÇÃO: SmartRetriever ===
        status_text.text("🔍 Analisando estrutura do documento...")
        progress_bar.progress(70)
        
        try:
            retriever = SmartRetriever()
            estrutura_info = retriever.initialize_with_document(documento)
            st.session_state['smart_retriever'] = retriever
            
            logger.info(f"SmartRetriever inicializado - Estrutura: {estrutura_info}")
            
            # Informar ao usuário sobre a estrutura encontrada
            if estrutura_info['capitulos'] > 0:
                st.sidebar.success(f"✅ {estrutura_info['capitulos']} capítulos identificados!")
            
            if estrutura_info['indice_encontrado']:
                st.sidebar.info("📋 Índice/Sumário detectado no documento")
                
        except Exception as e:
            logger.warning(f"SmartRetriever não pôde analisar estrutura: {e}")
            st.sidebar.warning("⚠️ Análise estrutural limitada")
        
        # Preparar contexto do sistema
        status_text.text("🤖 Configurando modelo de IA...")
        progress_bar.progress(80)
        
        # Obter mapa do documento se disponível
        mapa_documento = st.session_state.get('mapa_documento', '')
        
        if len(documento) > config.SMALL_DOCUMENT_THRESHOLD:
            st.session_state['usando_documento_grande'] = True
            # Para documentos grandes, usar estratégia de recuperação
            documento_preview = doc_manager.get_document_preview(max_chars=2000)
            
            system_message = f"""Você é um assistente especializado em análise de documentos.

Você tem acesso a um documento {tipo_arquivo} com as seguintes informações:

{mapa_documento if mapa_documento else f"- Total de páginas: {processamento['num_paginas']}\n- Tamanho: {len(documento)} caracteres\n- Processado em {processamento['total_chunks']} chunks"}

PREVIEW DO DOCUMENTO:
{documento_preview}

IMPORTANTE: Este é apenas um preview. Para cada pergunta do usuário, você receberá:
1. A estrutura completa do documento (capítulos, seções)
2. Os trechos mais relevantes do documento completo
3. Informações contextuais adicionais quando necessário

INSTRUÇÕES CRÍTICAS:
1. Use SEMPRE as informações dos trechos fornecidos para responder
2. Quando perguntarem sobre capítulos específicos, use o conteúdo COMPLETO fornecido
3. Se a informação não estiver nos trechos, diga "Não encontrei essa informação específica nos trechos analisados"
4. Cite números de página quando disponíveis
5. Seja preciso, detalhado e completo nas respostas
6. Para perguntas sobre estrutura (quantos capítulos, lista de capítulos), use o MAPA DO DOCUMENTO fornecido
7. Mantenha o contexto das perguntas anteriores quando relevante
8. Nunca invente informações - use apenas o que foi fornecido"""
        else:
            st.session_state['usando_documento_grande'] = False
            # Para documentos menores, incluir documento completo
            system_message = f"""Você é um assistente especializado em análise de documentos.

Você tem acesso completo ao seguinte documento {tipo_arquivo}:

====== DOCUMENTO COMPLETO ======
{documento}
====== FIM DO DOCUMENTO ======

{mapa_documento if mapa_documento else f"Total de páginas: {processamento['num_paginas']}\nTamanho: {len(documento)} caracteres"}

INSTRUÇÕES:
1. Use as informações do documento para responder às perguntas
2. Seja preciso, detalhado e completo
3. Cite números de página quando disponíveis
4. Se não encontrar a informação, seja honesto sobre isso
5. Mantenha o contexto das perguntas anteriores quando relevante
6. Nunca invente informações - use apenas o conteúdo do documento"""
        
        # Criar template do prompt
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
    memoria: ChatMessageHistory
) -> Generator[str, None, None]:
    """
    Processa perguntas usando chunks relevantes do documento de forma otimizada.
    VERSÃO MELHORADA com SmartRetriever.
    """
    try:
        memory_manager = st.session_state.get('doc_memory_manager')
        if not memory_manager:
            yield "❌ Erro: Sistema não conseguiu acessar o documento. Por favor, tente recarregar."
            return
        
        # Obter configurações
        k_chunks = st.session_state.get('k_chunks', config.DEFAULT_K_CHUNKS)
        usando_doc_grande = st.session_state.get('usando_documento_grande', False)
        
        # Usar SmartRetriever se disponível
        smart_retriever = st.session_state.get('smart_retriever')
        
        if not usando_doc_grande:
            # Documento pequeno - usar completo (já está no contexto do sistema)
            pergunta_completa = input_usuario
        else:
            # Documento grande - usar recuperação inteligente
            if smart_retriever and 'doc_chunks' in st.session_state:
                # === USAR SMART RETRIEVER ===
                chunks = st.session_state['doc_chunks']
                
                try:
                    chunks_relevantes, contexto_estrutural = smart_retriever.retrieve_with_structure(
                        input_usuario, chunks, k=k_chunks
                    )
                except Exception as e:
                    logger.error(f"Erro no SmartRetriever: {e}")
                    # Fallback para recuperação padrão
                    chunks_relevantes = memory_manager.retrieve_relevant_chunks(
                        input_usuario, k=k_chunks
                    )
                    contexto_estrutural = ""
            else:
                # Fallback para recuperação padrão
                chunks_relevantes = memory_manager.retrieve_relevant_chunks(
                    input_usuario, k=k_chunks * 2  # Pegar mais chunks
                )
                contexto_estrutural = ""
            
            if not chunks_relevantes:
                yield "⚠️ Não consegui encontrar informações relevantes no documento para responder sua pergunta. Tente reformular a pergunta ou seja mais específico."
                return
            
            # Montar contexto com os chunks
            contexto_chunks = []
            for i, chunk in enumerate(chunks_relevantes):
                chunk_info = f"[Trecho {i+1}"
                if 'chunk_id' in chunk.metadata:
                    chunk_info += f" - Chunk #{chunk.metadata['chunk_id']}"
                chunk_info += "]\n" + chunk.page_content
                contexto_chunks.append(chunk_info)
            
            contexto_relevante = "\n\n---\n\n".join(contexto_chunks)
            
            # Montar prompt final com todo o contexto
            if contexto_estrutural:
                prompt_adicional = f"""{contexto_estrutural}

TRECHOS RELEVANTES DO DOCUMENTO PARA ESTA PERGUNTA:
{contexto_relevante}


IMPORTANTE: 
- Use TODAS as informações acima (estrutura E trechos) para responder
- Se houver informações sobre capítulos específicos, use o conteúdo COMPLETO fornecido
- Seja detalhado e preciso
- Cite informações específicas dos trechos
- Se a resposta estiver clara nos trechos, forneça uma resposta completa"""
            else:
                prompt_adicional = f"""TRECHOS RELEVANTES DO DOCUMENTO PARA ESTA PERGUNTA:
{contexto_relevante}


IMPORTANTE:
- Use as informações acima para responder
- Seja preciso e detalhado
- Cite informações específicas quando possível"""
            
            pergunta_completa = f"{prompt_adicional}\n\nPERGUNTA DO USUÁRIO: {input_usuario}"
        
        # Debug info
        if st.session_state.get('show_debug', False):
            with st.expander("🔍 Debug - Informações de Processamento"):
                st.text(f"SmartRetriever ativo: {smart_retriever is not None}")
                st.text(f"Documento grande: {usando_doc_grande}")
                st.text(f"Tamanho do contexto: {len(pergunta_completa)} caracteres")
                st.text(f"Tokens estimados: ~{estimate_tokens(pergunta_completa)}")
                if usando_doc_grande and 'chunks_relevantes' in locals():
                    st.text(f"Chunks recuperados: {len(chunks_relevantes)}")
                    st.markdown("**Preview dos chunks:**")
                    for i, chunk in enumerate(chunks_relevantes[:3]):
                        st.text(f"\n--- Chunk {i+1} (ID: {chunk.metadata.get('chunk_id', 'N/A')}) ---")
                        st.code(chunk.page_content[:300] + "..." if len(chunk.page_content) > 300 else chunk.page_content)
                
                st.markdown("**Contexto completo enviado:**")
                st.code(pergunta_completa[:2000] + "..." if len(pergunta_completa) > 2000 else pergunta_completa)
        
        # Gerar resposta com streaming
        resposta_completa = ""
        for chunk in chain.stream({
            "input": pergunta_completa,
            "chat_history": memoria.messages
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
        
        # Log para análise
        logger.info(f"Pergunta processada - Tokens: {total_tokens}, Custo estimado: ${cost['total_estimated']:.4f}")
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {e}", exc_info=True)
        yield f"❌ Erro ao processar sua pergunta: {str(e)}\n\nTente reformular ou recarregar o documento."


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
                <p style='font-size: 1.1rem; color: #666;'>
                    Analise documentos com inteligência artificial de forma simples e poderosa.
                </p>
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
                   - O sistema identifica automaticamente capítulos e estrutura
                """)
            
            with st.expander("✨ Recursos Avançados"):
                st.markdown("""
                - 🎯 **Análise Precisa**: Respostas baseadas no conteúdo real do documento
                - 📊 **Identificação de Estrutura**: Detecta automaticamente capítulos e seções
                - 💬 **Conversação Natural**: Mantenha contexto entre perguntas
                - 🔍 **Busca Inteligente**: Sistema avançado de recuperação de informações
                - 📈 **Estatísticas**: Acompanhe uso de tokens e custos
                - 🛠️ **Diagnóstico**: Ferramenta para analisar e testar recuperação
                - 🎨 **Múltiplos Formatos**: Suporte para diversos tipos de arquivo
                """)
            
            with st.expander("💡 Dicas para melhores resultados"):
                st.markdown("""
                - Seja específico nas suas perguntas
                - Para documentos grandes, o sistema usa busca inteligente
                - Use o "Diagnóstico Avançado" para ver a estrutura detectada
                - Pergunte sobre capítulos específicos: "o que fala o primeiro capítulo?"
                - Aumente o número de chunks nas configurações avançadas para mais contexto
                """)
        
        st.stop()
    
    # Chat ativo
    memoria = st.session_state.get('memoria', ChatMessageHistory())
    
    # Container para mensagens
    chat_container = st.container()
    
    with chat_container:
        # Exibir histórico
        for mensagem in memoria.messages:
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
            with st.spinner("🤔 Analisando documento e preparando resposta..."):
                with chat_container:
                    resposta_container = st.empty()
                    resposta_completa = ""
                    
                    # Processar com streaming
                    for resposta_parcial in processar_pergunta_com_documento(
                        input_usuario, chain, memoria
                    ):
                        resposta_completa = resposta_parcial
                        resposta_container.markdown(
                            f'<div class="chat-message-ai">🤖 {resposta_completa}</div>',
                            unsafe_allow_html=True
                        )
            
            # Adicionar à memória
            memoria.add_user_message(input_usuario)
            memoria.add_ai_message(resposta_completa)
            st.session_state['memoria'] = memoria
            
        except Exception as e:
            with chat_container:
                st.error(f"❌ Erro ao processar resposta: {str(e)}")
            logger.error(f"Erro no chat: {e}", exc_info=True)


def sidebar():
    """Cria a barra lateral completa com todas as funcionalidades."""
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
            
            # Informações do SmartRetriever
            if 'estrutura_documento' in st.session_state:
                estrutura = st.session_state['estrutura_documento']
                st.text(f"• Capítulos: {len(estrutura.get('capitulos', []))}")
                st.text(f"• SmartRetriever: ✅ Ativo")
            
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
        
        # Opção de embeddings
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
            if st.session_state.get('documento_carregado', False):
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
            with st.spinner("⚙️ Processando documento..."):
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
    
    with col2:
        if st.button('🗑️ Limpar Chat', use_container_width=True):
            st.session_state['memoria'] = ChatMessageHistory()
            st.sidebar.success("✅ Chat limpo!")
    
    # Botão para novo documento
    if st.sidebar.button('📄 Novo Documento', use_container_width=True):
        # Limpar tudo relacionado ao documento
        keys_to_clear = [
            'chain', 'documento_completo', 'doc_memory_manager',
            'doc_chunks', 'vector_store', 'documento_carregado',
            'memoria', 'tamanho_documento', 'tipo_arquivo',
            'smart_retriever', 'estrutura_documento', 'mapa_documento'
        ]
        for key in keys_to_clear:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state['memoria'] = ChatMessageHistory()
        st.sidebar.success("✅ Pronto para novo documento!")
        st.rerun()
    
    # === DIAGNÓSTICO AVANÇADO ===
    st.sidebar.markdown("---")
    adicionar_interface_diagnostico()
    
    # === INFORMAÇÕES DO PROJETO ===
    st.sidebar.markdown("---")
    st.sidebar.caption("SOBRE")
    st.sidebar.info("""
    **Analyse Doc** v2.0
    
    Análise inteligente de documentos com IA
    
    ✨ SmartRetriever integrado
    🔍 Detecção automática de estrutura
    📊 Estatísticas em tempo real
    
    Desenvolvido com Streamlit + LangChain
    """)


def main():
    """Função principal da aplicação."""
    try:
        # Inicializar sessão
        inicializar_sessao()
        
        # Criar sidebar
        sidebar()
        
        # Página principal
        pagina_chat()
        
    except Exception as e:
        st.error(f"❌ Erro crítico na aplicação: {str(e)}")
        logger.error(f"Erro crítico: {e}", exc_info=True)
        
        if st.button("🔄 Reiniciar Aplicação"):
            st.session_state.clear()
            st.rerun()


if __name__ == '__main__':
    main()
