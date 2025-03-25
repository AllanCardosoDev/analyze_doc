import os
from time import sleep
import streamlit as st
from langchain_community.document_loaders import (
    WebBaseLoader,
    YoutubeLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
from fake_useragent import UserAgent
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

def carrega_site(url):
    """Carrega texto de um site usando WebBaseLoader."""
    documento = ""
    for i in range(5):
        try:
            os.environ["USER_AGENT"] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = "\n\n".join([doc.page_content for doc in lista_documentos])
            break
        except Exception as e:
            print(f"Erro ao carregar o site {i+1}: {e}")
            sleep(3)
    if not documento:
        return "⚠️ Não foi possível carregar o site."
    return documento

def carrega_youtube(video_id):
    """Carrega legendas de vídeos do YouTube."""
    try:
        loader = YoutubeLoader(video_id, add_video_info=False, language=["pt"])
        lista_documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        return f"❌ Erro ao carregar YouTube: {e}"

def carrega_csv(caminho):
    """Carrega dados de arquivos CSV."""
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    return "\n\n".join([doc.page_content for doc in lista_documentos])

def carrega_pdf(caminho):
    """Carrega e extrai texto de um PDF."""
    try:
        loader = PyPDFLoader(caminho)
        documentos = loader.load()
        texto = "\n\n".join([doc.page_content for doc in documentos])
        return texto if texto.strip() else "⚠️ O PDF não contém texto extraível."
    except Exception as e:
        return f"❌ Erro ao carregar PDF: {e}"

def carrega_txt(caminho):
    """Carrega e extrai texto de um arquivo TXT."""
    loader = TextLoader(caminho)
    lista_documentos = loader.load()
    return "\n\n".join([doc.page_content for doc in lista_documentos])

def carrega_docx(caminho):
    """Carrega e extrai texto de um arquivo DOCX."""
    try:
        doc = Document(caminho)
        texto_completo = []
        
        # Extrair texto de parágrafos
        for para in doc.paragraphs:
            if para.text.strip():
                texto_completo.append(para.text)
        
        # Extrair texto de tabelas
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    texto_completo.append(" | ".join(row_text))
        
        resultado = "\n\n".join(texto_completo)
        return resultado if resultado.strip() else "⚠️ O documento DOCX não contém texto extraível."
    except Exception as e:
        return f"❌ Erro ao carregar DOCX: {e}"

def gera_resumo(texto, max_length=1000):
    """
    Gera um resumo automático do texto usando chunking e extração de frases importantes.
    Esta é uma implementação básica. Para resultados melhores, use os modelos de LLM.
    """
    try:
        # Converter para documento Langchain
        doc = LangchainDocument(page_content=texto)
        
        # Dividir em chunks para processamento
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents([doc])
        
        # Extrair frases importantes (simplificado)
        frases_importantes = []
        for chunk in chunks:
            # Dividir em frases
            sentences = chunk.page_content.split('. ')
            # Selecionar as frases mais longas (geralmente mais informativas)
            sorted_sentences = sorted(sentences, key=len, reverse=True)
            # Pegar as 2-3 frases mais longas de cada chunk
            frases_importantes.extend(sorted_sentences[:min(3, len(sorted_sentences))])
        
        # Limitar o tamanho total do resumo
        resumo = ". ".join(frases_importantes)
        if len(resumo) > max_length:
            resumo = resumo[:max_length] + "..."
        
        return resumo
    except Exception as e:
        print(f"Erro ao gerar resumo: {e}")
        return "Não foi possível gerar um resumo automático."
