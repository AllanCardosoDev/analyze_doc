import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from loaders import *

TIPOS_ARQUIVOS_VALIDOS = ["Site", "Youtube", "Pdf", "Csv", "Txt"]

CONFIG_MODELOS = {
    "Groq": {
        "modelos": [
            "qwen-2.5-32b",
            "deepseek-r1-distill-qwen-32b",
            "deepseek-r1-distill-llama-70b-specdec",
            "deepseek-r1-distill-llama-70b",
            "llama-3.3-70b-specdec",
            "llama-3.2-1b-preview",
            "llama-3.2-3b-preview",
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
        ],
        "chat": ChatGroq,
    },
    "OpenAI": {
        "modelos": ["gpt-4o-mini", "gpt-4o", "o1-preview", "o1-mini"],
        "chat": ChatOpenAI,
    },
}

MEMORIA = ConversationBufferMemory()

def carrega_arquivos(tipo_arquivo, arquivo):
    if not arquivo:
        return "Nenhum arquivo foi fornecido."

    if tipo_arquivo == "Site":
        return carrega_site(arquivo)
    elif tipo_arquivo == "Youtube":
        return carrega_youtube(arquivo)
    elif tipo_arquivo in ["Pdf", "Csv", "Txt"]:
        with tempfile.NamedTemporaryFile(suffix=f".{tipo_arquivo.lower()}", delete=False) as temp:
            temp.write(arquivo.read())
            return carrega_pdf(temp.name) if tipo_arquivo == "Pdf" else (
                carrega_csv(temp.name) if tipo_arquivo == "Csv" else carrega_txt(temp.name)
            )
    return "Tipo de arquivo inválido."

def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):
    if not api_key:
        st.error("⚠️ API Key não fornecida. Adicione uma chave válida para continuar.")
        return
    
    documento = carrega_arquivos(tipo_arquivo, arquivo)
    system_message = f"""
    Você é um assistente amigável chamado Oráculo.
    Utilize as informações abaixo do documento ({tipo_arquivo}) para basear suas respostas:
    
    ###
    {documento}
    ###
    Sempre que houver "$" na saída, substitua por "S".
    Se o documento contiver "Just a moment...Enable JavaScript and cookies to continue", peça para o usuário tentar novamente.
    """

    template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("placeholder", "{chat_history}"),
        ("user", "{input}")
    ])
    
    chat = CONFIG_MODELOS[provedor]["chat"](model=modelo, api_key=api_key)
    chain = template | chat
    st.session_state["chain"] = chain

def pagina_chat():
    st.header("🤖 Bem-vindo ao Oráculo", divider=True)

    chain = st.session_state.get("chain")
    if chain is None:
        st.error("⚠️ Nenhum modelo carregado. Por favor, carregue o Oráculo.")
        return

    memoria = st.session_state.get("memoria", MEMORIA)
    for mensagem in memoria.buffer_as_messages:
        st.chat_message(mensagem.type).markdown(mensagem.content)

    input_usuario = st.chat_input("Fale com o Oráculo")
    if input_usuario:
        st.chat_message("human").markdown(input_usuario)

        try:
            resposta = chain.stream({
                "input": input_usuario,
                "chat_history": memoria.buffer_as_messages
            })
            st.chat_message("ai").write_stream(resposta)
            memoria.chat_memory.add_user_message(input_usuario)
            memoria.chat_memory.add_ai_message(resposta)
            st.session_state["memoria"] = memoria

        except Exception as e:
            st.error(f"❌ Erro ao processar resposta: {e}")

def sidebar():
    tabs = st.tabs(["Upload de Arquivos", "Seleção de Modelos"])

    with tabs[0]:
        tipo_arquivo = st.selectbox("Selecione o tipo de arquivo", TIPOS_ARQUIVOS_VALIDOS)
        arquivo = None
        if tipo_arquivo in ["Site", "Youtube"]:
            arquivo = st.text_input(f"Digite a URL do {tipo_arquivo.lower()}")
        else:
            arquivo = st.file_uploader(f"Faça o upload do arquivo {tipo_arquivo.lower()}", type=[tipo_arquivo.lower()])

    with tabs[1]:
        provedor = st.selectbox("Selecione o provedor do modelo", list(CONFIG_MODELOS.keys()))
        modelo = st.selectbox("Selecione o modelo", CONFIG_MODELOS[provedor]["modelos"])
        api_key = st.text_input(f"Adicione a API key para {provedor}", type="password")

    if st.button("Inicializar Oráculo"):
        carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo)

    if st.button("Apagar Histórico de Conversa"):
        st.session_state["memoria"] = MEMORIA

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()

if __name__ == "__main__":
    main()
