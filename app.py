import tempfile
import streamlit as st
import os
import json
import uuid
from datetime import datetime
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import *
from visualizers import *
from processors import *
from security import *
from collaborative import *

# Configuração inicial da página
st.set_page_config(
    page_title="Analyse Doc",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tipos de arquivos suportados
TIPOS_ARQUIVOS_VALIDOS = [
    "Site", "Youtube", "Pdf", "Docx", "Csv", "Txt", 
    "Xlsx", "Pptx", "Json", "Md", "Html", "Imagem"
]

# Configuração dos modelos de IA
CONFIG_MODELOS = {
    "Groq": {
        "modelos": [
            "distil-whisper-large-v3-en",
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "llama-guard-3-8b",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "mixtral-8x7b-32768",
            "whisper-large-v3",
            "whisper-large-v3-turbo",
        ],
        "chat": ChatGroq,
        "parametros": {
            "temperature": 0.7,
            "max_tokens": 4000
        }
    },
    "OpenAI": {
        "modelos": ["gpt-4o-mini", "gpt-4o", "o1-preview", "o1-mini"],
        "chat": ChatOpenAI,
        "parametros": {
            "temperature": 0.7,
            "max_tokens": 4000
        }
    },
    "Anthropic": {
        "modelos": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
        "chat": None,  # Placeholder para implementação futura
        "parametros": {
            "temperature": 0.7,
            "max_tokens": 4000
        }
    },
    "Local": {
        "modelos": ["llama3-local", "mistral-local"],
        "chat": None,  # Placeholder para implementação futura
        "parametros": {
            "temperature": 0.7,
            "max_tokens": 4000
        }
    }
}

# Inicialização da memória de conversa
MEMORIA = ConversationBufferMemory()

# Inicialização de variáveis de sessão
if "user_id" not in st.session_state:
    st.session_state["user_id"] = str(uuid.uuid4())
if "documents" not in st.session_state:
    st.session_state["documents"] = []
if "current_document" not in st.session_state:
    st.session_state["current_document"] = None
if "document_analyses" not in st.session_state:
    st.session_state["document_analyses"] = {}
if "shared_users" not in st.session_state:
    st.session_state["shared_users"] = []
if "comments" not in st.session_state:
    st.session_state["comments"] = []
if "document_metadata" not in st.session_state:
    st.session_state["document_metadata"] = {}
if "visualizations" not in st.session_state:
    st.session_state["visualizations"] = {}

def carrega_arquivos(tipo_arquivo, arquivo, configuracoes=None):
    """Função para carregar arquivos com tratamento de erros e opções avançadas."""
    if configuracoes is None:
        configuracoes = {}
    
    try:
        # Extração de metadados comuns
        file_id = str(uuid.uuid4())
        metadata = {
            "id": file_id,
            "tipo": tipo_arquivo,
            "data_processamento": datetime.now().isoformat(),
            "processado_por": st.session_state["user_id"],
            "configuracoes": configuracoes
        }
        
        if tipo_arquivo == "Site":
            conteudo = carrega_site(arquivo)
            metadata["url"] = arquivo
        elif tipo_arquivo == "Youtube":
            proxy = st.session_state.get("youtube_proxy", None)
            conteudo = carrega_youtube(arquivo, proxy=proxy)
            metadata["url"] = arquivo
        elif tipo_arquivo == "Pdf":
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_pdf(temp.name, ocr=configuracoes.get("usar_ocr", False))
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        elif tipo_arquivo == "Docx":
            with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_docx(temp.name)
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        elif tipo_arquivo == "Csv":
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_csv(temp.name)
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        elif tipo_arquivo == "Txt":
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_txt(temp.name)
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        elif tipo_arquivo == "Xlsx":
            with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_xlsx(temp.name)
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        elif tipo_arquivo == "Pptx":
            with tempfile.NamedTemporaryFile(suffix=".pptx", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_pptx(temp.name)
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        elif tipo_arquivo == "Json":
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_json(temp.name)
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        elif tipo_arquivo == "Md":
            with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_markdown(temp.name)
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        elif tipo_arquivo == "Html":
            with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_html(temp.name)
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        elif tipo_arquivo == "Imagem":
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
                temp.write(arquivo.read())
                conteudo = carrega_imagem(temp.name, configuracoes.get("nivel_ocr", "simples"))
                metadata["nome_arquivo"] = arquivo.name
                metadata["tamanho"] = arquivo.size
        
        # Guardamos os metadados para uso posterior
        st.session_state["document_metadata"][file_id] = metadata
        
        # Se anonimização estiver ativada, aplicamos
        if configuracoes.get("anonimizar", False):
            conteudo = anonimizar_texto(conteudo)
            
        # Armazenamos o documento na sessão
        st.session_state["current_document"] = {
            "id": file_id,
            "content": conteudo,
            "metadata": metadata
        }
        
        return conteudo
    except Exception as e:
        return f"❌ Erro ao carregar arquivo: {e}"

def processa_documento(documento, configuracoes):
    """Aplica processamentos avançados ao documento."""
    try:
        documento_processado = documento
        doc_id = st.session_state["current_document"]["id"]
        analises = {}
        
        # Gerar resumo se solicitado
        if configuracoes.get("gerar_resumo", False):
            max_length = configuracoes.get("max_resumo_length", 1000)
            resumo = gera_resumo(documento_processado, max_length)
            documento_processado = resumo
            analises["resumo"] = resumo
        
        # Traduzir se necessário
        idioma_codigo = configuracoes.get("idioma_codigo", "pt")
        if idioma_codigo != "pt" and configuracoes.get("traduzir", False):
            documento_processado = traduz_texto(documento_processado, idioma_codigo)
            analises["traducao"] = {
                "idioma_original": "detectado",
                "idioma_alvo": idioma_codigo,
                "texto_traduzido": documento_processado
            }
        
        # Extrair entidades se solicitado
        if configuracoes.get("extrair_entidades", False):
            entidades = extrai_entidades(documento)
            analises["entidades"] = entidades
        
        # Análise de sentimento
        if configuracoes.get("analise_sentimento", False):
            sentimento = analisa_sentimento(documento)
            analises["sentimento"] = sentimento
        
        # Detecção de tópicos
        if configuracoes.get("detectar_topicos", False):
            topicos = detecta_topicos(documento)
            analises["topicos"] = topicos
        
        # Classificação de documento
        if configuracoes.get("classificar_documento", False):
            classificacao = classifica_documento(documento)
            analises["classificacao"] = classificacao
        
        # Verificação de fatos
        if configuracoes.get("verificar_fatos", False):
            fatos = verifica_fatos(documento)
            analises["fatos_verificados"] = fatos
        
        # Detecção de viés
        if configuracoes.get("detectar_vies", False):
            vies = detecta_vies(documento)
            analises["vies"] = vies
        
        # Extração de tabelas
        if configuracoes.get("extrair_tabelas", False):
            tabelas = extrai_tabelas(documento)
            analises["tabelas"] = tabelas
        
        # Extração de dados financeiros
        if configuracoes.get("extrair_dados_financeiros", False):
            dados_financeiros = extrai_dados_financeiros(documento)
            analises["dados_financeiros"] = dados_financeiros
        
        # Extração de fórmulas
        if configuracoes.get("extrair_formulas", False):
            formulas = extrai_formulas(documento)
            analises["formulas"] = formulas
        
        # Extração de referências
        if configuracoes.get("extrair_referencias", False):
            referencias = extrai_referencias(documento)
            analises["referencias"] = referencias
            
        # Criar visualizações
        if configuracoes.get("criar_visualizacoes", False):
            visualizacoes = {}
            
            if configuracoes.get("nuvem_palavras", False):
                nuvem = gera_nuvem_palavras(documento)
                visualizacoes["nuvem_palavras"] = nuvem
            
            if configuracoes.get("grafico_entidades", False) and "entidades" in analises:
                grafico = gera_grafico_entidades(analises["entidades"])
                visualizacoes["grafico_entidades"] = grafico
                
            if configuracoes.get("mapa_conexoes", False):
                mapa = gera_mapa_conexoes(documento)
                visualizacoes["mapa_conexoes"] = mapa
                
            if configuracoes.get("timeline", False):
                timeline = gera_timeline(documento)
                visualizacoes["timeline"] = timeline
                
            # Armazenar visualizações em sessão para uso posterior
            st.session_state["visualizations"][doc_id] = visualizacoes
            
        # Armazenar todas as análises para uso posterior
        st.session_state["document_analyses"][doc_id] = analises
        
        return documento_processado
    except Exception as e:
        st.error(f"Erro ao processar documento: {e}")
        return documento

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo, configuracoes=None):
    """Carrega o modelo de IA e prepara o sistema para responder com base no documento."""
    if configuracoes is None:
        configuracoes = {}
        
    if not api_key:
        st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
        return
    
    # Carregar documento
    documento = carrega_arquivos(tipo_arquivo, arquivo, configuracoes)
    if documento.startswith("❌") or documento.startswith("⚠️"):
        st.error(documento)
        return
    
    # Processar documento com as configurações avançadas
    documento_processado = processa_documento(documento, configuracoes)
    
    # Opções de estilo de resposta
    estilo_resposta = configuracoes.get("estilo_resposta", "equilibrado")
    nivel_detalhe = configuracoes.get("nivel_detalhe", "médio")
    formato_resposta = configuracoes.get("formato_resposta", "narrativo")
    persona = configuracoes.get("persona", "assistente")
    
    # Construir o prompt do sistema com base nas opções
    system_message = f"""
    Você é um assistente chamado Analyse Doc, especializado em análise de documentos.
    
    Estilo de resposta: {estilo_resposta}
    Nível de detalhe: {nivel_detalhe}
    Formato: {formato_resposta}
    Persona: {persona}
    
    Aqui está o conteúdo do documento ({tipo_arquivo}) carregado:
    ###
    {documento_processado[:10000]}  # Aumentamos o limite para documentos maiores
    ###
    
    Responda com base nesse conteúdo.
    Se não conseguir acessar alguma informação solicitada, informe ao usuário.
    """
    
    # Adicionar informações de análises avançadas, se disponíveis
    doc_id = st.session_state["current_document"]["id"]
    if doc_id in st.session_state["document_analyses"]:
        analises = st.session_state["document_analyses"][doc_id]
        if analises:
            system_message += "\n\nInformações adicionais da análise automática:\n"
            
            if "entidades" in analises:
                system_message += f"\nEntidades principais: {', '.join(analises['entidades'][:10])}\n"
                
            if "topicos" in analises:
                system_message += f"\nTópicos principais: {', '.join(analises['topicos'][:5])}\n"
                
            if "classificacao" in analises:
                system_message += f"\nTipo de documento: {analises['classificacao']}\n"
                
            if "sentimento" in analises:
                system_message += f"\nSentimento geral: {analises['sentimento']}\n"
    
    # Configurações do modelo
    modelo_params = CONFIG_MODELOS[provedor].get("parametros", {}).copy()
    
    # Sobrescrever com configurações personalizadas se fornecidas
    if "temperature" in configuracoes:
        modelo_params["temperature"] = configuracoes["temperature"]
    if "max_tokens" in configuracoes:
        modelo_params["max_tokens"] = configuracoes["max_tokens"]
    
    # Criar o template do chat
    template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])
    
    # Instanciar o modelo de chat com os parâmetros
    chat_class = CONFIG_MODELOS[provedor]["chat"]
    chat = chat_class(model=modelo, api_key=api_key, **modelo_params)
    
    # Criar a cadeia de processamento
    chain = template | chat
    
    # Salvar na sessão
    st.session_state["chain"] = chain

def pagina_chat():
    """Cria a interface do chat e gerencia a conversa do usuário."""
    # Interface dividida em duas colunas
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🤖 Analyse Doc", divider=True)
        
        chain = st.session_state.get("chain")
        if chain is None:
            st.error("Carregue o Analyse Doc primeiro.")
            st.info("Use a barra lateral para carregar um documento e inicializar o sistema.")
            st.stop()
        
        # Interface de chat
        memoria = st.session_state.get("memoria", MEMORIA)
        
        # Container para histórico de chat com altura fixa
        chat_container = st.container(height=500)
        with chat_container:
            for mensagem in memoria.buffer_as_messages:
                st.chat_message(mensagem.type).markdown(mensagem.content)
        
        # Entrada de chat
        input_usuario = st.chat_input("Fale com o Analyse Doc")
        if input_usuario:
            st.chat_message("human").markdown(input_usuario)
            
            # Configuração de streaming para respostas mais fluidas
            with st.chat_message("ai"):
                resposta_container = st.empty()
                resposta_completa = ""
                
                # Streaming da resposta
                for chunk in chain.stream({
                    "input": input_usuario,
                    "chat_history": memoria.buffer_as_messages
                }):
                    resposta_completa += chunk
                    resposta_container.markdown(resposta_completa + "▌")
                
                # Atualização final sem cursor
                resposta_container.markdown(resposta_completa)
            
            # Adicionar à memória
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta_completa)
            st.session_state["memoria"] = memoria
            
            # Adicionar aos comentários se estiver em modo colaborativo
            if st.session_state.get("modo_colaborativo", False):
                novo_comentario = {
                    "id": str(uuid.uuid4()),
                    "documento_id": st.session_state["current_document"]["id"],
                    "usuario_id": st.session_state["user_id"],
                    "tipo": "pergunta",
                    "texto": input_usuario,
                    "resposta": resposta_completa,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state["comments"].append(novo_comentario)
    
    # Painel de informações e visualizações
    with col2:
        if st.session_state["current_document"]:
            doc_id = st.session_state["current_document"]["id"]
            metadata = st.session_state["document_metadata"].get(doc_id, {})
            
            st.subheader("📊 Informações do Documento")
            
            # Metadados básicos
            if "nome_arquivo" in metadata:
                st.info(f"Arquivo: {metadata['nome_arquivo']}")
            elif "url" in metadata:
                st.info(f"URL: {metadata['url']}")
            
            st.caption(f"Tipo: {metadata.get('tipo', 'Desconhecido')}")
            st.caption(f"Processado em: {metadata.get('data_processamento', 'Desconhecido')}")
            
            # Abas para diferentes visualizações
            tabs = st.tabs(["Análises", "Visualizações", "Colaboração"])
            
            with tabs[0]:
                analises = st.session_state["document_analyses"].get(doc_id, {})
                if analises:
                    if "entidades" in analises:
                        st.subheader("Entidades")
                        entidades_str = ", ".join(analises["entidades"][:15])
                        st.write(entidades_str)
                    
                    if "topicos" in analises:
                        st.subheader("Tópicos")
                        for topico in analises["topicos"][:5]:
                            st.write(f"• {topico}")
                    
                    if "sentimento" in analises:
                        st.subheader("Sentimento")
                        st.write(analises["sentimento"])
                    
                    if "classificacao" in analises:
                        st.subheader("Classificação")
                        st.write(analises["classificacao"])
                else:
                    st.info("Nenhuma análise avançada disponível. Configure na aba 'Processamento'.")
            
            with tabs[1]:
                visualizacoes = st.session_state["visualizations"].get(doc_id, {})
                if visualizacoes:
                    if "nuvem_palavras" in visualizacoes:
                        st.subheader("Nuvem de Palavras")
                        st.image(visualizacoes["nuvem_palavras"], use_column_width=True)
                    
                    if "grafico_entidades" in visualizacoes:
                        st.subheader("Gráfico de Entidades")
                        st.pyplot(visualizacoes["grafico_entidades"])
                    
                    if "mapa_conexoes" in visualizacoes:
                        st.subheader("Mapa de Conexões")
                        st.plotly_chart(visualizacoes["mapa_conexoes"], use_container_width=True)
                    
                    if "timeline" in visualizacoes:
                        st.subheader("Timeline")
                        st.plotly_chart(visualizacoes["timeline"], use_container_width=True)
                else:
                    st.info("Nenhuma visualização disponível. Ative 'Criar visualizações' na aba 'Processamento'.")
            
            with tabs[2]:
                # Ativar/desativar modo colaborativo
                st.session_state["modo_colaborativo"] = st.toggle("Modo Colaborativo", 
                                                               value=st.session_state.get("modo_colaborativo", False))
                
                if st.session_state["modo_colaborativo"]:
                    # Adicionar usuários para compartilhamento
                    with st.expander("Compartilhar"):
                        novo_usuario = st.text_input("Email do usuário")
                        if st.button("Adicionar"):
                            if novo_usuario and novo_usuario not in st.session_state["shared_users"]:
                                st.session_state["shared_users"].append(novo_usuario)
                                st.success(f"Documento compartilhado com {novo_usuario}")
                    
                    # Lista de usuários com acesso
                    if st.session_state["shared_users"]:
                        st.subheader("Compartilhado com")
                        for usuario in st.session_state["shared_users"]:
                            cols = st.columns([3, 1])
                            cols[0].write(usuario)
                            if cols[1].button("Remover", key=f"rm_{usuario}"):
                                st.session_state["shared_users"].remove(usuario)
                                st.rerun()
                    
                    # Comentários e anotações
                    st.subheader("Comentários")
                    comentarios_doc = [c for c in st.session_state["comments"] 
                                      if c["documento_id"] == doc_id]
                    
                    if comentarios_doc:
                        for comentario in comentarios_doc:
                            with st.expander(f"{comentario['timestamp'][:10]} - {comentario['tipo'].capitalize()}"):
                                st.write(comentario["texto"])
                                if "resposta" in comentario:
                                    st.write("**Resposta:**")
                                    st.write(comentario["resposta"])
                    
                    # Adicionar anotação/comentário manual
                    with st.form("novo_comentario"):
                        tipo_comentario = st.selectbox("Tipo", ["comentário", "anotação", "tarefa"])
                        texto_comentario = st.text_area("Texto")
                        submit = st.form_submit_button("Adicionar")
                        
                        if submit and texto_comentario:
                            novo_comentario = {
                                "id": str(uuid.uuid4()),
                                "documento_id": doc_id,
                                "usuario_id": st.session_state["user_id"],
                                "tipo": tipo_comentario,
                                "texto": texto_comentario,
                                "timestamp": datetime.now().isoformat()
                            }
                            st.session_state["comments"].append(novo_comentario)
                            st.success("Comentário adicionado!")
                            st.rerun()
                else:
                    st.info("Ative o Modo Colaborativo para compartilhar este documento e adicionar comentários.")

def sidebar():
    """Cria a barra lateral para upload de arquivos e seleção de modelos."""
    with st.sidebar:
        st.image("https://via.placeholder.com/150x150.png?text=AD", width=150)
        st.title("Analyse Doc")
        
        tabs = st.tabs(["Arquivos", "Seleção de Modelos", "Processamento", "Visualização", 
                      "Segurança", "Configurações", "Exportação"])
        
        with tabs[0]:  # Arquivos
            st.subheader("Upload de Arquivos")
            tipo_arquivo = st.selectbox("Selecione o tipo de arquivo", TIPOS_ARQUIVOS_VALIDOS)
            
            # Interface diferente para URLs vs uploads
            if tipo_arquivo in ["Site", "Youtube"]:
                arquivo = st.text_input(f"Digite a URL do {tipo_arquivo.lower()}")
            else:
                # Mapear extensões de arquivo para cada tipo
                extensoes = {
                    "Pdf": ["pdf"],
                    "Docx": ["docx", "doc"],
                    "Csv": ["csv"],
                    "Txt": ["txt"],
                    "Xlsx": ["xlsx", "xls"],
                    "Pptx": ["pptx", "ppt"],
                    "Json": ["json"],
                    "Md": ["md"],
                    "Html": ["html", "htm"],
                    "Imagem": ["jpg", "jpeg", "png", "gif", "bmp"]
                }
                
                arquivo = st.file_uploader(
                    f"Faça o upload do arquivo {tipo_arquivo.lower()}", 
                    type=extensoes.get(tipo_arquivo, [tipo_arquivo.lower()])
                )
            
            # Pré-visualização se estiver disponível
            if arquivo and tipo_arquivo not in ["Site", "Youtube"]:
                with st.expander("Pré-visualização"):
                    if tipo_arquivo == "Pdf":
                        st.warning("Pré-visualização de PDF não disponível")
                    elif tipo_arquivo == "Csv" or tipo_arquivo == "Xlsx":
                        st.dataframe(arquivo)
                    elif tipo_arquivo == "Imagem":
                        st.image(arquivo)
                    elif tipo_arquivo in ["Txt", "Json", "Md", "Html"]:
                        st.code(arquivo.read().decode('utf-8'), language='text')
            
            # Histórico de documentos
            if st.session_state["documents"]:
                with st.expander("Histórico de Documentos"):
                    for i, doc in enumerate(st.session_state["documents"]):
                        meta = st.session_state["document_metadata"].get(doc["id"], {})
                        nome = meta.get("nome_arquivo", meta.get("url", f"Documento {i+1}"))
                        if st.button(f"{nome}", key=f"hist_{doc['id']}"):
                            st.session_state["current_document"] = doc
                            st.success(f"Documento '{nome}' carregado!")
                            st.rerun()
        
        with tabs[1]:  # Seleção de Modelos
            st.subheader("Modelos de IA")
            provedor = st.selectbox("Selecione o provedor do modelo", list(CONFIG_MODELOS.keys()))
            modelo = st.selectbox("Selecione o modelo", CONFIG_MODELOS[provedor]["modelos"])
            api_key = st.text_input(f"Adicione a API key para {provedor}", type="password")
            
            # Configurações avançadas do modelo
            with st.expander("Configurações avançadas do modelo"):
                temperatura = st.slider("Temperatura", 0.0, 1.0, 0.7,
                                      help="Valores mais altos geram respostas mais criativas")
                max_tokens = st.slider("Máximo de tokens", 500, 8000, 4000,
                                     help="Limite de tamanho da resposta")
                
                st.session_state["model_config"] = {
                    "temperature": temperatura,
                    "max_tokens": max_tokens
                }
        
        with tabs[2]:  # Processamento
            st.subheader("Processamento avançado")
            
            # Resumo automático
            st.checkbox("Gerar resumo automático", key="gerar_resumo",
                       help="Cria um resumo do documento antes de processar")
            
            st.slider("Comprimento máximo do resumo", 500, 8000, 5000, key="max_resumo_length",
                    help="Número máximo de caracteres no resumo")
            
            # Idioma e tradução
            idiomas = {
                "Português": "pt", "Inglês": "en", "Espanhol": "es", "Francês": "fr",
                "Alemão": "de", "Italiano": "it", "Japonês": "ja", "Chinês": "zh"
            }
            
            col1, col2 = st.columns(2)
            with col1:
                idioma_selecionado = st.selectbox("Idioma de saída", list(idiomas.keys()),
                                                key="idioma_saida",
                                                help="Idioma das respostas")
                st.session_state["idioma_codigo"] = idiomas[idioma_selecionado]
            
            with col2:
                st.checkbox("Traduzir documento", key="traduzir",
                          help="Traduz o documento para o idioma selecionado")
            
            # Análise de texto
            cols = st.columns(2)
            with cols[0]:
                st.checkbox("Extrair entidades", key="extrair_entidades",
                          help="Identifica nomes, organizações e outras entidades")
                
                st.checkbox("Análise de sentimento", key="analise_sentimento",
                          help="Analisa o tom emocional do documento")
                
                st.checkbox("Detecção de tópicos", key="detectar_topicos",
                          help="Identifica os temas principais do documento")
                
                st.checkbox("Classificar documento", key="classificar_documento",
                          help="Identifica o tipo/categoria do documento")
            
            with cols[1]:
                st.checkbox("Verificar fatos", key="verificar_fatos", disabled=True,
                          help="Verifica afirmações contra fontes confiáveis (em breve)")
                
                st.checkbox("Detectar viés", key="detectar_vies", disabled=True,
                          help="Identifica linguagem tendenciosa (em breve)")
                
                st.checkbox("OCR para imagens/PDFs", key="usar_ocr",
                          help="Extrai texto de imagens e PDFs escaneados")
                
                st.selectbox("Nível de OCR", ["simples", "avançado", "completo"],
                           key="nivel_ocr", help="Qualidade da extração de texto de imagens")
            
            # Extração estruturada
            with st.expander("Extração estruturada"):
                col1, col2 = st.columns(2)
                with col1:
                    st.checkbox("Extrair tabelas", key="extrair_tabelas",
                              help="Identifica e extrai tabelas do documento")
                    
                    st.checkbox("Extrair dados financeiros", key="extrair_dados_financeiros",
                              help="Identifica valores monetários, porcentagens, etc.")
                
                with col2:
                    st.checkbox("Extrair fórmulas", key="extrair_formulas", disabled=True,
                              help="Identifica equações e fórmulas (em breve)")
                    
                    st.checkbox("Extrair referências", key="extrair_referencias", disabled=True,
                              help="Compila citações e referências (em breve)")
        
        with tabs[3]:  # Visualização
            st.subheader("Visualizações")
            
            st.checkbox("Criar visualizações", key="criar_visualizacoes",
                       help="Gera visualizações automáticas baseadas no documento")
            
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Nuvem de palavras", key="nuvem_palavras",
                          help="Cria representação visual das palavras mais frequentes")
                
                st.checkbox("Gráfico de entidades", key="grafico_entidades",
                          help="Visualiza as entidades extraídas")
            
            with col2:
                st.checkbox("Mapa de conexões", key="mapa_conexoes",
                          help="Mostra relacionamentos entre conceitos")
                
                st.checkbox("Timeline", key="timeline", disabled=True,
                          help="Cria linha do tempo para documentos com elementos temporais (em breve)")
            
            # Configuração de gráficos
            with st.expander("Configurações de visualização"):
                st.selectbox("Estilo de gráficos", 
                           ["streamlit", "plotly", "seaborn", "minimalista", "corporativo"],
                           key="estilo_graficos")
                
                st.color_picker("Cor primária", "#1E88E5", key="cor_primaria")
                st.color_picker("Cor secundária", "#FF5722", key="cor_secundaria")
        
        with tabs[4]:  # Segurança
            st.subheader("Segurança e Privacidade")
            
            st.checkbox("Anonimizar texto", key="anonimizar",
                       help="Remove ou mascara informações pessoais sensíveis")
            
            st.selectbox("Nível de anonimização", 
                       ["básico", "intermediário", "completo"],
                       key="nivel_anonimizacao",
                       help="Determina quais tipos de dados serão anonimizados")
            
            st.selectbox("Retenção de dados", 
                       ["sessão atual", "7 dias", "30 dias", "permanente"],
                       key="retencao_dados",
                       help="Por quanto tempo os dados serão armazenados")
            
            st.checkbox("Criptografar documento", key="criptografar", disabled=True,
                       help="Criptografa o documento na sessão (em breve)")
            
            st.checkbox("Registrar logs de acesso", key="registrar_logs", disabled=True,
                       help="Mantém registro de quem acessou o documento (em breve)")
        
        with tabs[5]:  # Configurações
            st.subheader("Configurações do Sistema")
            
            # Configurações do YouTube
            with st.expander("Configurações do YouTube"):
                proxy = st.text_input(
                    "Proxy para YouTube (formato: http://usuario:senha@host:porta)",
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
            
            # Personalização de resposta
            with st.expander("Personalização de resposta"):
                st.selectbox("Estilo de resposta", 
                           ["equilibrado", "acadêmico", "simplificado", "técnico", "conversacional"],
                           key="estilo_resposta")
                
                st.selectbox("Nível de detalhe", 
                           ["mínimo", "baixo", "médio", "alto", "máximo"],
                           key="nivel_detalhe")
                
                st.selectbox("Formato das respostas", 
                           ["narrativo", "estruturado", "ponto a ponto", "tabular"],
                           key="formato_resposta")
                
                st.selectbox("Persona do assistente", 
                           ["assistente", "especialista", "professor", "colega", "consultor"],
                           key="persona")
            
            # Preferências de interface
            with st.expander("Preferências de interface"):
                theme = st.selectbox("Tema", ["Claro", "Escuro", "Automático"], key="theme")
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
                
                layout = st.selectbox("Layout", ["Padrão", "Wide", "Centrado"], key="layout")
                if layout == "Wide":
                    st.markdown('<style>.reportview-container .main .block-container{max-width: 1200px; padding: 1rem;}</style>', unsafe_allow_html=True)
                elif layout == "Centrado":
                    st.markdown('<style>.reportview-container .main .block-container{max-width: 800px; padding: 2rem; margin: auto;}</style>', unsafe_allow_html=True)
        
        with tabs[6]:  # Exportação
            st.subheader("Exportação e Compartilhamento")
            
            formato_export = st.selectbox("Formato de exportação", 
                                        ["PDF", "DOCX", "HTML", "Markdown", "JSON", "TXT"])
            
            col1, col2 = st.columns(2)
            with col1:
                st.checkbox("Incluir análises", key="export_analises", value=True)
                st.checkbox("Incluir metadados", key="export_metadados", value=True)
            
            with col2:
                st.checkbox("Incluir visualizações", key="export_visualizacoes", value=True)
                st.checkbox("Incluir comentários", key="export_comentarios", value=False)
            
            if st.button("Exportar análise", use_container_width=True):
                if st.session_state["current_document"]:
                    with st.spinner("Preparando exportação..."):
                        # Placeholder para funcionalidade real
                        doc_id = st.session_state["current_document"]["id"]
                        nome_doc = st.session_state["document_metadata"].get(doc_id, {}).get(
                            "nome_arquivo", "documento")
                        nome_arquivo = f"analyse-doc_{nome_doc}_{datetime.now().strftime('%Y%m%d')}"
                        
                        st.success(f"Arquivo '{nome_arquivo}.{formato_export.lower()}' pronto para download!")
                        # Aqui seria implementado o download real
                else:
                    st.error("Nenhum documento carregado para exportar.")
            
            # Opções de compartilhamento
            st.subheader("Compartilhamento")
            
            metodo_compartilhamento = st.selectbox("Método de compartilhamento", 
                                                 ["Link", "Email", "Código QR"])
            
            if st.button("Gerar compartilhamento", use_container_width=True):
                if st.session_state["current_document"]:
                    # Placeholder para funcionalidade real
                    st.success(f"Compartilhamento por {metodo_compartilhamento} gerado!")
                    if metodo_compartilhamento == "Link":
                        st.code("https://example.com/analyse-doc/share/abc123")
                    elif metodo_compartilhamento == "Código QR":
                        st.image("https://via.placeholder.com/150x150.png?text=QR", width=150)
                else:
                    st.error("Nenhum documento carregado para compartilhar.")
        
        # Botões principais
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Inicializar Analyse Doc", use_container_width=True):
                # Coletar todas as configurações
                configuracoes = {
                    # Modelo
                    **st.session_state.get("model_config", {}),
                    
                    # Processamento
                    "gerar_resumo": st.session_state.get("gerar_resumo", False),
                    "max_resumo_length": st.session_state.get("max_resumo_length", 1000),
                    "idioma_codigo": st.session_state.get("idioma_codigo", "pt"),
                    "traduzir": st.session_state.get("traduzir", False),
                    "extrair_entidades": st.session_state.get("extrair_entidades", False),
                    "analise_sentimento": st.session_state.get("analise_sentimento", False),
                    "detectar_topicos": st.session_state.get("detectar_topicos", False),
                    "classificar_documento": st.session_state.get("classificar_documento", False),
                    "verificar_fatos": st.session_state.get("verificar_fatos", False),
                    "detectar_vies": st.session_state.get("detectar_vies", False),
                    "usar_ocr": st.session_state.get("usar_ocr", False),
                    "nivel_ocr": st.session_state.get("nivel_ocr", "simples"),
                    "extrair_tabelas": st.session_state.get("extrair_tabelas", False),
                    "extrair_dados_financeiros": st.session_state.get("extrair_dados_financeiros", False),
                    "extrair_formulas": st.session_state.get("extrair_formulas", False),
                    "extrair_referencias": st.session_state.get("extrair_referencias", False),
                    
                    # Visualização
                    "criar_visualizacoes": st.session_state.get("criar_visualizacoes", False),
                    "nuvem_palavras": st.session_state.get("nuvem_palavras", False),
                    "grafico_entidades": st.session_state.get("grafico_entidades", False),
                    "mapa_conexoes": st.session_state.get("mapa_conexoes", False),
                    "timeline": st.session_state.get("timeline", False),
                    "estilo_graficos": st.session_state.get("estilo_graficos", "streamlit"),
                    "cor_primaria": st.session_state.get("cor_primaria", "#1E88E5"),
                    "cor_secundaria": st.session_state.get("cor_secundaria", "#FF5722"),
                    
                    # Segurança
                    "anonimizar": st.session_state.get("anonimizar", False),
                    "nivel_anonimizacao": st.session_state.get("nivel_anonimizacao", "básico"),
                    "retencao_dados": st.session_state.get("retencao_dados", "sessão atual"),
                    "criptografar": st.session_state.get("criptografar", False),
                    "registrar_logs": st.session_state.get("registrar_logs", False),
                    
                    # Personalização
                    "estilo_resposta": st.session_state.get("estilo_resposta", "equilibrado"),
                    "nivel_detalhe": st.session_state.get("nivel_detalhe", "médio"),
                    "formato_resposta": st.session_state.get("formato_resposta", "narrativo"),
                    "persona": st.session_state.get("persona", "assistente"),
                }
                
                # Inicializar modelo
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo, configuracoes)
                
                # Armazenar documento na lista de histórico
                if st.session_state["current_document"] and st.session_state["current_document"] not in st.session_state["documents"]:
                    st.session_state["documents"].append(st.session_state["current_document"])
        
        with col2:
            if st.button("Apagar Histórico de Conversa", use_container_width=True):
                st.session_state["memoria"] = MEMORIA
                st.success("Histórico de conversa apagado!")

def main():
    """Função principal do aplicativo."""
    sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
