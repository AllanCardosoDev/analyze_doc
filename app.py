import os
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import carrega_site, carrega_youtube, carrega_pdf, carrega_docx, carrega_csv, carrega_txt
from loaders import gera_resumo, gera_pdf_resumo, traduz_texto
from resumo import gerar_resumo_documento
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
    
    /* Destacar a opção de resumo PDF */
    .pdf-option {
        background-color: #e8f5e9;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
        border-left: 3px solid #43a047;
    }
    
    /* Ícones com animação */
    .animated-icon {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.7; }
        50% { opacity: 1; }
        100% { opacity: 0.7; }
    }
    
    /* Barra de carregamento personalizada */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    
    /* Tooltip personalizado */
    .tooltip {
        position: relative;
        display: inline-block;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 120px;
        background-color: #555;
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -60px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Novos estilos para o resumo automático */
    .resumo-panel {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 20px;
        border-left: 4px solid #1976d2;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    
    .resumo-panel h3 {
        color: #1976d2;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    
    .meta-info {
        display: flex;
        flex-wrap: wrap;
        gap: 15px;
        margin-bottom: 15px;
        font-size: 0.9rem;
    }
    
    .meta-info span {
        background-color: #bbdefb;
        padding: 5px 10px;
        border-radius: 15px;
    }
    
    .resumo-content {
        background-color: white;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
    }
    
    .resumo-actions {
        display: flex;
        gap: 10px;
    }
    
    /* Estilos para a seção de resumo automático */
    .auto-resumo-container {
        background-color: #f5f5f5;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid #e0e0e0;
    }
    
    .auto-resumo-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 10px;
    }
    
    .auto-resumo-content {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #e0e0e0;
        margin-bottom: 15px;
    }
    
    .auto-resumo-secao {
        margin-bottom: 15px;
    }
    
    .auto-resumo-secao h4 {
        color: #1976d2;
        margin-bottom: 8px;
        font-size: 1.1rem;
    }
    
    .separador {
        height: 1px;
        background-color: #e0e0e0;
        margin: 15px 0;
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

def gerar_resumo_automatico(documento, tipo_arquivo, modelo, provedor, api_key):
    """
    Gera um resumo completo do documento utilizando o modelo de IA.
    
    Args:
        documento (str): O texto do documento
        tipo_arquivo (str): Tipo do arquivo (PDF, DOCX, etc.)
        modelo (str): Nome do modelo de IA
        provedor (str): Nome do provedor (Groq, OpenAI)
        api_key (str): Chave de API para o modelo
    
    Returns:
        dict: Dicionário com o resumo e metadados
    """
    try:
        # Iniciar o modelo de IA
        chat = CONFIG_MODELOS[provedor]["chat"](model=modelo, api_key=api_key)
        
        # Configurações para o resumo
        max_length = st.session_state.get("max_resumo_length", 1500)
        idioma_codigo = st.session_state.get("idioma_codigo", "pt")
        
        # Dicionário de configurações para o resumo
        config = {
            "max_length": max_length,
            "idioma": idioma_codigo,
            "usar_llm": st.session_state.get("usar_llm_resumo", True),
            "llm_chain": chat,
            "tradutor_disponivel": st.session_state.get("tradutor_disponivel", False),
            "incluir_topicos": st.session_state.get("incluir_topicos", True),
            "incluir_termos": st.session_state.get("incluir_termos", True),
            "analisar_estrutura": st.session_state.get("analisar_estrutura", False)
        }
        
        # Gerar o resumo do documento
        resultado_resumo = gerar_resumo_documento(documento, tipo_arquivo, config)
        
        return resultado_resumo
    except Exception as e:
        st.error(f"Erro ao gerar o resumo automático: {e}")
        return None

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
        st.session_state["tipo_arquivo"] = tipo_arquivo
        
        # Guardar metadados do documento para referência
        st.session_state["documento_meta"] = {
            "tipo": tipo_arquivo,
            "tamanho": len(documento),
            "data_processamento": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "modelo": modelo,
            "provedor": provedor
        }
        
        # Gerar resumo automático do documento se solicitado
        if st.session_state.get("gerar_resumo", False):
            with st.spinner("Gerando resumo automático do documento..."):
                resultado_resumo = gerar_resumo_automatico(
                    documento, tipo_arquivo, modelo, provedor, api_key
                )
                
                if resultado_resumo:
                    st.session_state["resumo_documento"] = resultado_resumo["resumo"]
                    st.session_state["resumo_pdf_bytes"] = resultado_resumo["pdf_bytes"]
                    st.session_state["resumo_filename"] = f"resumo_{tipo_arquivo}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                    st.session_state["resumo_seccoes"] = resultado_resumo.get("seccoes", {})
                    st.success("✅ Resumo automático gerado com sucesso!")
        
        st.success(f"✅ Modelo {modelo} carregado com sucesso! Documento pronto para análise.")
    except Exception as e:
        st.error(f"❌ Erro ao carregar o modelo: {e}")

def create_download_link(pdf_bytes, filename):
    """Cria um link para download do PDF"""
    b64 = base64.b64encode(pdf_bytes).decode()
    # Estilo do botão de download para ficar mais atrativo
    href = f'''
    <a href="data:application/pdf;base64,{b64}" 
       download="{filename}" 
       style="
           display: inline-block;
           background-color: #1E88E5;
           color: white;
           padding: 10px 20px;
           text-align: center;
           text-decoration: none;
           font-weight: bold;
           border-radius: 5px;
           margin: 10px 0;
           width: 100%;
           transition: all 0.3s;
           box-shadow: 0 2px 4px rgba(0,0,0,0.2);
       "
       onmouseover="this.style.backgroundColor='#1565C0'; this.style.boxShadow='0 4px 8px rgba(0,0,0,0.3)';"
       onmouseout="this.style.backgroundColor='#1E88E5'; this.style.boxShadow='0 2px 4px rgba(0,0,0,0.2)';"
    >
        📥 Baixar Resumo em PDF
    </a>
    '''
    return href

def exibir_resumo_automatico():
    """Exibe o resumo automático completo do documento."""
    if "resumo_documento" in st.session_state:
        meta = st.session_state.get("documento_meta", {})
        resumo = st.session_state["resumo_documento"]
        resumo_seccoes = st.session_state.get("resumo_seccoes", {})
        
        st.markdown('<div class="auto-resumo-container">', unsafe_allow_html=True)
        
        # Cabeçalho do resumo
        st.markdown('<div class="auto-resumo-header">', unsafe_allow_html=True)
        st.markdown('### 📄 Resumo Automático do Documento')
        
        # Se temos um PDF gerado, mostrar botão de download
        if "resumo_pdf_bytes" in st.session_state and "resumo_filename" in st.session_state:
            download_link = create_download_link(
                st.session_state["resumo_pdf_bytes"], 
                st.session_state["resumo_filename"]
            )
            st.markdown(download_link, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Metadados do documento
        st.markdown('<div class="meta-info">', unsafe_allow_html=True)
        st.markdown(f'<span>Tipo: {meta.get("tipo", "Desconhecido")}</span>', unsafe_allow_html=True)
        st.markdown(f'<span>Processado em: {meta.get("data_processamento", "")}</span>', unsafe_allow_html=True)
        st.markdown(f'<span>Tamanho: {meta.get("tamanho", 0)} caracteres</span>', unsafe_allow_html=True)
        st.markdown(f'<span>Modelo: {meta.get("modelo", "")} ({meta.get("provedor", "")})</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Resumo principal
        st.markdown('<div class="auto-resumo-secao">', unsafe_allow_html=True)
        st.markdown('<h4>📝 Síntese Geral</h4>', unsafe_allow_html=True)
        st.markdown('<div class="auto-resumo-content">', unsafe_allow_html=True)
        st.markdown(resumo)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Exibir seções específicas se disponíveis
        if resumo_seccoes:
            st.markdown('<div class="separador"></div>', unsafe_allow_html=True)
            st.markdown('<div class="auto-resumo-secao">', unsafe_allow_html=True)
            st.markdown('<h4>🔍 Análise por Seções</h4>', unsafe_allow_html=True)
            
            for titulo, conteudo in resumo_seccoes.items():
                st.markdown(f'<h5>{titulo}</h5>', unsafe_allow_html=True)
                st.markdown('<div class="auto-resumo-content">', unsafe_allow_html=True)
                st.markdown(conteudo)
                st.markdown('</div>', unsafe_allow_html=True)
                
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Nenhum resumo disponível. Selecione a opção 'Gerar resumo automático' antes de inicializar o modelo.")

def pagina_chat():
    """Cria a interface do chat e gerencia a conversa do usuário."""
    st.markdown('<h1 class="main-header">📑 Analyse Doc</h1>', unsafe_allow_html=True)
    
    # Verificar se temos um resumo do documento para exibir
    if "mostrar_resumo" in st.session_state and st.session_state["mostrar_resumo"]:
        exibir_resumo_automatico()
    
    # Exibir informações do documento se disponível
    if "documento_meta" in st.session_state:
        meta = st.session_state["documento_meta"]
        with st.container():
            st.markdown('<div class="document-info">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Tipo:** {meta['tipo']}")
                st.markdown(f"**Tamanho:** {meta['tamanho']} caracteres")
                st.markdown(f"**Modelo:** {meta['modelo']}")
                st.markdown(f"**Provedor:** {meta['provedor']}")
                st.markdown(f"**Processado em:** {meta['data_processamento']}")
            
            with col2:
                st.markdown('<div class="pdf-section">', unsafe_allow_html=True)
                st.markdown('<h4>📄 Ações do Documento</h4>', unsafe_allow_html=True)
                
                if st.button("📋 Mostrar/Ocultar Resumo Completo", key="btn_toggle_resumo"):
                    st.session_state["mostrar_resumo"] = not st.session_state.get("mostrar_resumo", False)
                    st.experimental_rerun()
                
                # Verificar se já temos um PDF gerado do resumo durante a inicialização
                if "resumo_pdf_bytes" in st.session_state and "resumo_filename" in st.session_state:
                    download_link = create_download_link(
                        st.session_state["resumo_pdf_bytes"], 
                        st.session_state["resumo_filename"]
                    )
                    st.markdown(download_link, unsafe_allow_html=True)
                else:
                    # Botão para gerar resumo em PDF
                    if st.button("📥 Gerar Resumo em PDF", key="btn_gerar_pdf", use_container_width=True):
                        with st.spinner("Gerando resumo em PDF..."):
                            try:
                                # Obter o texto completo do documento
                                documento = st.session_state.get("documento_completo", "")
                                
                                # Definir o comprimento máximo do resumo
                                max_length = st.session_state.get("max_resumo_length", 1000)
                                
                                # Gerar o resumo
                                resumo = gera_resumo(documento, max_length)
                                
                                # Gerar o PDF
                                pdf_bytes = gera_pdf_resumo(
                                    resumo, 
                                    meta['tipo'], 
                                    meta['data_processamento']
                                )
                                
                                # Criar link de download
                                filename = f"resumo_{meta['tipo']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                                download_link = create_download_link(pdf_bytes, filename)
                                st.markdown(download_link, unsafe_allow_html=True)
                                
                                # Guardar o resumo na sessão para uso posterior
                                st.session_state["resumo_documento"] = resumo
                                st.session_state["resumo_pdf_bytes"] = pdf_bytes
                                st.session_state["resumo_filename"] = filename
                                
                            except Exception as e:
                                st.error(f"Erro ao gerar o PDF: {e}")
                                st.error(f"Detalhes: {traceback.format_exc()}")
                
                st.caption("O resumo em PDF contém os principais pontos do documento analisado, formatados para leitura e compartilhamento.")
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    chain = st.session_state.get("chain")
    if chain is None:
        st.info("📚 Carregue um documento e inicialize o Analyse Doc para começar.")
        
        # Adicionar demonstração visual
        st.markdown("""
        <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;">
            <img src="https://cdn.pixabay.com/photo/2016/12/23/07/01/documents-1926996_960_720.png" width="200">
            <h3>Como usar o Analyse Doc</h3>
            <ol style="text-align: left; display: inline-block;">
                <li>Na barra lateral, selecione a aba <b>Upload de Arquivos</b></li>
                <li>Escolha o tipo de documento e faça o upload</li>
                <li>Na aba <b>Seleção de Modelos</b>, escolha o provedor e o modelo</li>
                <li>Clique em <b>Inicializar Analyse Doc</b></li>
                <li>Comece a fazer perguntas sobre o documento</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        
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
    tabs = st.sidebar.tabs(["Upload de Arquivos", "Seleção de Modelos", "Processamento", "Configurações"])
    
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
        st.markdown("### ⚙️ Processamento avançado")
        
        # Modificar o texto do checkbox para deixar claro sobre o resumo automático
        st.checkbox(
            "Gerar resumo automático", 
            key="gerar_resumo", 
            help="Cria um resumo detalhado do documento assim que for carregado",
            value=True  # Ativado por padrão
        )
        
        # Adicionar opção para usar LLM para resumo
        st.checkbox(
            "Usar IA para melhorar resumo", 
            key="usar_llm_resumo", 
            help="Utiliza o modelo de IA para gerar um resumo mais elaborado (recomendado)",
            value=True  # Ativado por padrão
        )
        
        # Adicionar um ícone visual de PDF ao lado do slider
        col1, col2 = st.columns([5, 1])
        with col1:
            st.slider(
                "Comprimento máximo do resumo", 
                500, 5000, 1500, 
                key="max_resumo_length",
                help="Número máximo de caracteres no resumo"
            )
        with col2:
            st.markdown("📄", help="Tamanho do resumo em caracteres")
        
        idiomas = {"Português": "pt", "Inglês": "en", "Espanhol": "es", "Francês": "fr"}
        idioma_selecionado = st.selectbox(
            "Idioma de saída", 
            list(idiomas.keys()), 
            key="idioma_saida",
            help="Traduzir o conteúdo para este idioma"
        )
        st.session_state["idioma_codigo"] = idiomas[idioma_selecionado]
        
        # Verificar se a tradução está disponível (exige um modelo de LLM)
        st.session_state["tradutor_disponivel"] = False  # Por padrão, não disponível
        
        # Novas opções de resumo
        st.markdown('<div class="pdf-option">', unsafe_allow_html=True)
        st.markdown("#### 📊 Configurações do Resumo")
        
        st.checkbox(
            "Incluir tópicos principais", 
            key="incluir_topicos", 
            value=True,
            help="Identifica e lista os tópicos principais do documento"
        )
        
        st.checkbox(
            "Incluir análise de termos-chave", 
            key="incluir_termos", 
            value=True,
            help="Extrai os termos mais relevantes do documento"
        )
        
        st.checkbox(
            "Analisar estrutura do documento", 
            key="analisar_estrutura", 
            help="Identifica seções e a estrutura geral do documento"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.checkbox(
            "Extrair entidades", 
            key="extrair_entidades", 
            disabled=True,
            help="Identifica nomes, organizações e outras entidades (em breve)"
        )
        
        st.checkbox(
            "Análise de sentimento", 
            key="analise_sentimento", 
            disabled=True,
            help="Analisa o tom emocional do documento (em breve)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
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
                # Verificar se devemos processar o documento
                if st.session_state.get("gerar_resumo", False) and arquivo:
                    documento = carrega_arquivos(tipo_arquivo, arquivo)
                    if not (isinstance(documento, str) and (documento.startswith("❌") or documento.startswith("⚠️"))):
                        # Iniciar o modelo e gerar resumo automático
                        st.session_state["mostrar_resumo"] = True
                        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)
                else:
                    # Iniciar o modelo normalmente
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

