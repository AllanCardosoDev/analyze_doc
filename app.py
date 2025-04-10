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
    carrega_youtube,
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

# Aplicar estilo padrão
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 600;
        color: #4F8BF9;
        text-align: center;
        margin-bottom: 1rem;
    }
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
    'Site', 'Youtube', 'Pdf', 'Docx', 'Csv', 'Txt'
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
        elif tipo_arquivo == "Youtube":
            return carrega_youtube(arquivo)
        
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
        # Limite reduzido para 25K caracteres para economia de tokens
        limite_tamanho = 25000
        
        # Processamos todos os documentos com o gerenciador de memória para ter acesso ao número de páginas
        memory_manager = st.session_state['doc_memory_manager']
        processamento = memory_manager.process_document(documento, tipo_arquivo)
        
        # Dependendo do tamanho do documento, usamos abordagens diferentes
        if len(documento) > limite_tamanho:
            # Para documentos muito grandes
            st.session_state['usando_documento_grande'] = True
            # Obter um preview do documento para o contexto inicial (1000 caracteres apenas)
            documento_preview = memory_manager.get_document_preview(max_chars=1000)
            # Informar o usuário sobre o uso do método para documentos grandes
            st.sidebar.info(f"📄 Documento grande ({len(documento)} caracteres, ~{processamento['num_paginas']} páginas) - Usando processamento avançado.")
            
            # Mensagem do sistema mais concisa
            system_message = f"""Você é um assistente especializado em analisar documentos.
            Tipo: {tipo_arquivo} | Tamanho: {len(documento)} caracteres | ~{processamento['num_paginas']} páginas
            
            Preview do documento (você tem acesso ao documento completo via sistema de recuperação):
            {documento_preview}
            
            Seja detalhado e preciso em suas respostas, sempre usando as informações do documento.
            """
        else:
            # Para documentos menores
            st.session_state['usando_documento_grande'] = False
            
            # Ainda usamos recuperação por chunks para economizar tokens
            st.sidebar.success(f"📄 Documento processado ({len(documento)} caracteres, ~{processamento['num_paginas']} páginas)")
            
            # Mensagem do sistema
            system_message = f"""Você é um assistente especializado em analisar documentos.
            
            SOBRE O DOCUMENTO:
            - Tipo: {tipo_arquivo}
            - Tamanho: {len(documento)} caracteres
            - Páginas estimadas: {processamento['num_paginas']}
            
            Utilize as informações do documento para responder às perguntas.
            Seja preciso e objetivo em suas análises.
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
        
        # Avisa o usuário que o documento foi carregado com sucesso
        st.sidebar.success(f"✅ Documento {tipo_arquivo} carregado com sucesso!")
        
    except Exception as e:
        logger.error(f"Erro ao carregar modelo: {e}")
        st.error(f"❌ Erro ao processar documento: {e}")

def processar_pergunta_documento_grande(input_usuario, chain):
    """
    Processa perguntas para documentos com recuperação de contexto.
    Otimizado para usar menos tokens.
    """
    try:
        # Verificar se a pergunta é sobre o número de páginas
        import re
        if re.search(r'quantas\s+p[áa]ginas|n[úu]mero\s+de\s+p[áa]ginas', input_usuario.lower()):
            num_paginas = st.session_state.get('num_paginas', 0)
            if num_paginas > 0:
                yield f"O documento possui aproximadamente {num_paginas} páginas."
                return
                
        # Obter o gerenciador de memória de documentos
        memory_manager = st.session_state.get('doc_memory_manager')
        if not memory_manager:
            yield "Erro: Gerenciador de memória de documentos não inicializado."
            return
            
        # Obter o número de chunks configurado pelo usuário (padrão 2)
        k_chunks = st.session_state.get('k_chunks', 2)
        
        # Recuperar chunks relevantes para a pergunta
        chunks_relevantes = memory_manager.retrieve_relevant_chunks(input_usuario, k=k_chunks)
        
        # Se recebermos um chunk especial com informação de páginas, usamos diretamente
        if chunks_relevantes and hasattr(chunks_relevantes[0], 'metadata') and 'num_paginas' in chunks_relevantes[0].metadata:
            yield chunks_relevantes[0].page_content
            return
            
        # Combinar o conteúdo dos chunks relevantes
        contexto_relevante = "\n\n".join([chunk.page_content for chunk in chunks_relevantes])
        
        # Criar um prompt específico para esta pergunta (otimizado para tokens)
        prompt_especifico = f"""
        Responda usando estas informações do documento:
        {contexto_relevante}
        
        Pergunta: {input_usuario}
        """
        
        # Usar o chain para gerar a resposta
        resposta = ""
        for chunk in chain.stream({"input": prompt_especifico}):
            if hasattr(chunk, 'content'):
                resposta += chunk.content
            else:
                resposta += str(chunk)
            yield resposta
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {e}")
        yield f"Erro ao processar sua pergunta: {e}"

def pagina_chat():
    """Interface principal do chat."""
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
    
    # Cria container para o chat
    chat_container = st.container()
    with chat_container:
        # Exibe o histórico de mensagens
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
                    
                    # Sempre usar a abordagem de recuperação de contexto
                    # Esta é a abordagem mais econômica em tokens
                    for resposta_parcial in processar_pergunta_documento_grande(input_usuario, chain):
                        resposta_container.markdown(
                            f'<div class="chat-message-ai">{resposta_parcial}</div>',
                            unsafe_allow_html=True
                        )
                    resposta_completa = resposta_parcial
            
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
        elif tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a URL do vídeo', placeholder="https://www.youtube.com/watch?v=...")
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
        
        # Exibir informações do documento atual se disponível
        if 'doc_memory_manager' in st.session_state and 'documento_completo' in st.session_state:
            memory_manager = st.session_state['doc_memory_manager']
            info = memory_manager.get_document_info()
            st.markdown("**Informações do documento atual:**")
            st.text(f"• Tipo: {info['tipo']}")
            st.text(f"• Tamanho: {info['tamanho']} caracteres")
            st.text(f"• Páginas estimadas: {info['num_paginas']}")
            st.text(f"• Chunks processados: {info['num_chunks']}")
            
            # Opção para ajustar o tamanho dos chunks (para usuários avançados)
            st.caption("Ajustes avançados")
            chunk_size = st.slider(
                "Tamanho dos chunks (caracteres)",
                min_value=1000,
                max_value=4000,
                value=2000,
                step=500,
                help="Chunks menores usam menos tokens mas podem perder contexto"
            )
            
            # Opção para ajustar o número de chunks retornados
            k_chunks = st.slider(
                "Número de chunks por consulta",
                min_value=1,
                max_value=4,
                value=2,
                step=1,
                help="Mais chunks fornecem mais contexto mas usam mais tokens"
            )
            
            # Guardar configurações na sessão
            st.session_state['chunk_size'] = chunk_size
            st.session_state['k_chunks'] = k_chunks
    
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
        num_paginas = st.session_state.get('num_paginas', 0)
        if num_paginas > 0:
            st.sidebar.info(f"📄 {st.session_state['tipo_arquivo']} • {st.session_state['tamanho_documento']} caracteres • ~{num_paginas} páginas")
        else:
            st.sidebar.info(f"📄 {st.session_state['tipo_arquivo']} • {st.session_state['tamanho_documento']} caracteres")
        
        # Mostrar modo de processamento
        if st.session_state.get('usando_documento_grande', False):
            st.sidebar.success("🔄 Usando processamento avançado para documento grande")
        else:
            st.sidebar.info("🔄 Usando processamento padrão")
    
    # Informações do projeto
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
