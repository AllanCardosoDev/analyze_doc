import tempfile
import time
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, VideoUnavailable, NoTranscriptFound

# Modelos disponíveis
CONFIG_MODELOS = {
    "Groq": {
        "modelos": ["llama-3.1-70b-versatile", "gemma2-9b-it", "mixtral-8x7b-32768"],
        "chat": ChatGroq
    },
    "OpenAI": {
        "modelos": ["gpt-4o-mini", "gpt-4o", "o1-preview", "o1-mini"],
        "chat": ChatOpenAI
    }
}

TIPOS_ARQUIVOS_VALIDOS = ["Site", "Youtube", "Pdf", "Csv", "Txt"]
MEMORIA = ConversationBufferMemory()

def carrega_youtube(video_url):
    """Carrega transcrição do YouTube e trata erros"""
    try:
        video_id = video_url.split("v=")[1].split("&")[0]
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return "\n".join([entry["text"] for entry in transcript])
    except (TranscriptsDisabled, VideoUnavailable, NoTranscriptFound) as e:
        return f"Erro ao carregar transcrição: {str(e)}"
    except Exception as e:
        return f"Erro desconhecido ao carregar o vídeo: {str(e)}"

def carrega_arquivos(tipo_arquivo, arquivo):
    """Carrega arquivos dependendo do tipo"""
    if tipo_arquivo == "Site":
        return "Carregamento de site não implementado"
    elif tipo_arquivo == "Youtube":
        return carrega_youtube(arquivo)
    elif tipo_arquivo in ["Pdf", "Csv", "Txt"]:
        with tempfile.NamedTemporaryFile(suffix=f'.{tipo_arquivo.lower()}', delete=False) as temp:
            temp.write(arquivo.read())
            return f"Arquivo {tipo_arquivo} carregado."
    return "Tipo de arquivo inválido"

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    """Inicializa o modelo e prepara as respostas"""
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    system_message = f"Você é um assistente chamado Oráculo. Utilize as informações do documento {tipo_arquivo}: {documento}"

    template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])

    chat = CONFIG_MODELOS[provedor]["chat"](model=modelo, api_key=api_key)
    chain = template | chat

    st.session_state["chain"] = chain

def pagina_chat():
    """Interface do chatbot"""
    st.header("🤖 Bem-vindo ao Oráculo", divider=True)

    chain = st.session_state.get("chain")
    if chain is None:
        st.error("Carregue o Oráculo antes de começar.")
        st.stop()

    memoria = st.session_state.get("memoria", MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    input_usuario = st.chat_input("Fale com o Oráculo")
    if input_usuario:
        st.chat_message("human").markdown(input_usuario)

        try:
            time.sleep(1)  # Evita limite de requisições
            resposta = chain.stream({
                "input": input_usuario,
                "chat_history": memoria.buffer_as_messages
            })

            st.chat_message("ai").write_stream(resposta)
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta)
            st.session_state["memoria"] = memoria

        except Exception as e:
            st.error(f"Erro ao processar a resposta: {e}")

def sidebar():
    """Barra lateral com upload de arquivos e seleção de modelos"""
    tabs = st.tabs(["Upload de Arquivos", "Seleção de Modelos"])

    with tabs[0]:
        tipo_arquivo = st.selectbox("Selecione o tipo de arquivo", TIPOS_ARQUIVOS_VALIDOS)
        arquivo = None
        if tipo_arquivo == "Youtube":
            arquivo = st.text_input("Digite a URL do vídeo")
        elif tipo_arquivo in ["Pdf", "Csv", "Txt"]:
            arquivo = st.file_uploader(f"Faça o upload do arquivo {tipo_arquivo.lower()}", type=[tipo_arquivo.lower()])

    with tabs[1]:
        provedor = st.selectbox("Selecione o provedor do modelo", CONFIG_MODELOS.keys())
        modelo = st.selectbox("Selecione o modelo", CONFIG_MODELOS[provedor]["modelos"])
        api_key = st.text_input(f"Adicione a API key para {provedor}", type="password")

    if st.button("Inicializar Oráculo"):
        if not api_key:
            st.warning("⚠️ Insira sua chave da API para continuar.")
            st.stop()
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

    if st.button("Apagar Histórico de Conversa"):
        st.session_state["memoria"] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
