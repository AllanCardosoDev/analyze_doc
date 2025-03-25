"""
Módulo de carregadores para diferentes tipos de documentos.
Fornece funções para extrair texto de arquivos e conteúdo web.
"""

import os
from time import sleep
import re
import streamlit as st
from langchain_community.document_loaders import (
    WebBaseLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
from fake_useragent import UserAgent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

# Importações condicionais para lidar com dependências opcionais
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

def carrega_site(url):
    """
    Carrega texto de um site usando WebBaseLoader.
    
    Args:
        url: URL do site a ser carregado
        
    Returns:
        String com o conteúdo do site ou mensagem de erro
    """
    documento = ""
    for i in range(3):  # Tenta até 3 vezes
        try:
            # Usa um User-Agent aleatório para evitar bloqueios
            os.environ["USER_AGENT"] = UserAgent().random
            
            # Carrega o site
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            
            # Concatena o conteúdo de todas as páginas
            documento = "\n\n".join([doc.page_content for doc in lista_documentos])
            break
        except Exception as e:
            print(f"Tentativa {i+1}: Erro ao carregar o site: {e}")
            sleep(2)  # Espera 2 segundos antes de tentar novamente
    
    if not documento:
        return "⚠️ Não foi possível carregar o site. Verifique a URL e tente novamente."
    
    return documento

def carrega_youtube(video_id, proxy=None):
    """
    Carrega legendas de vídeos do YouTube com suporte a proxy.
    
    Args:
        video_id: ID ou URL do vídeo do YouTube
        proxy: String de proxy no formato 'http://usuário:senha@host:porta'
        
    Returns:
        String com as legendas do vídeo ou mensagem de erro
    """
    try:
        # Extrai o video_id de uma URL completa, se for fornecida
        if "youtube.com" in video_id or "youtu.be" in video_id:
            if "youtube.com/watch?v=" in video_id:
                video_id = video_id.split("youtube.com/watch?v=")[1].split("&")[0]
            elif "youtu.be/" in video_id:
                video_id = video_id.split("youtu.be/")[1].split("?")[0]
        
        # Importa diretamente o youtube_transcript_api para mais controle
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            from youtube_transcript_api.formatters import TextFormatter
        except ImportError:
            return "❌ Biblioteca youtube_transcript_api não encontrada. Instale-a com: pip install youtube_transcript_api"
        
        # Configuração de proxy, se fornecido
        proxies = None
        if proxy:
            proxies = {'http': proxy, 'https': proxy}
        
        # Tenta obter as legendas em diferentes idiomas
        languages = ['pt', 'pt-BR', 'en']
        
        # Obtém as legendas diretamente, com suporte a proxy
        if proxies:
            try:
                from youtube_transcript_api import TranscriptListFetcher
                
                fetcher = TranscriptListFetcher(proxies=proxies)
                transcript_list = fetcher.fetch_transcript_list(video_id)
                transcript = transcript_list.find_transcript(languages)
                transcripts = transcript.fetch()
            except Exception as e:
                return f"❌ Erro ao buscar legendas via proxy: {e}"
        else:
            # Método padrão sem proxy
            try:
                transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            except Exception as e:
                return f"❌ Erro ao buscar legendas: {e}"
        
        # Verifica se encontrou legendas
        if not transcripts:
            return "⚠️ Não foi possível encontrar legendas para este vídeo."
        
        # Combina os segmentos de legendas em um texto formatado
        texto_completo = ""
        for entry in transcripts:
            texto_completo += f"{entry.get('text', '')} "
        
        return texto_completo
    
    except Exception as e:
        mensagem_erro = str(e)
        if "IP" in mensagem_erro and "block" in mensagem_erro:
            return """❌ O YouTube está bloqueando as requisições do seu IP. Isso pode acontecer por:
            
1. Muitas requisições foram feitas em um curto período
2. Você está usando um IP de um provedor de nuvem (AWS, Google Cloud, etc.)

Soluções:
- Configure um proxy na aba 'Configurações'
- Espere algumas horas e tente novamente
- Use uma VPN ou outra rede para acessar
            """
        return f"❌ Erro ao carregar YouTube: {e}"

def carrega_csv(caminho):
    """
    Carrega dados de arquivos CSV.
    
    Args:
        caminho: Caminho para o arquivo CSV
        
    Returns:
        String com o conteúdo do CSV formatado
    """
    try:
        loader = CSVLoader(caminho)
        lista_documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        return f"❌ Erro ao carregar CSV: {e}"

def carrega_pdf(caminho):
    """
    Carrega e extrai texto de um PDF.
    
    Args:
        caminho: Caminho para o arquivo PDF
        
    Returns:
        String com o conteúdo do PDF
    """
    try:
        loader = PyPDFLoader(caminho)
        documentos = loader.load()
        texto = "\n\n".join([doc.page_content for doc in documentos])
        return texto if texto.strip() else "⚠️ O PDF não contém texto extraível."
    except Exception as e:
        return f"❌ Erro ao carregar PDF: {e}"

def carrega_txt(caminho):
    """
    Carrega e extrai texto de um arquivo TXT.
    
    Args:
        caminho: Caminho para o arquivo TXT
        
    Returns:
        String com o conteúdo do arquivo TXT
    """
    try:
        loader = TextLoader(caminho)
        lista_documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        return f"❌ Erro ao carregar TXT: {e}"

def carrega_docx(caminho):
    """
    Carrega e extrai texto de um arquivo DOCX.
    
    Args:
        caminho: Caminho para o arquivo DOCX
        
    Returns:
        String com o conteúdo do documento DOCX
    """
    if not DOCX_AVAILABLE:
        return "❌ Biblioteca python-docx não encontrada. Instale-a com: pip install python-docx"
    
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
    
    Args:
        texto: Texto a ser resumido
        max_length: Tamanho máximo do resumo em caracteres
        
    Returns:
        String com o resumo gerado
    """
    try:
        # Verificar se o texto já é curto o suficiente
        if len(texto) <= max_length:
            return texto
        
        # Converter para documento Langchain
        doc = LangchainDocument(page_content=texto)
        
        # Dividir em chunks para processamento
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents([doc])
        
        # Extrair frases importantes
        frases_importantes = []
        
        for chunk in chunks:
            # Dividir em frases
            sentences = chunk.page_content.split('. ')
            
            # Filtrar frases muito curtas
            sentences = [s for s in sentences if len(s.split()) > 5]
            
            # Ordenar frases por comprimento (uma heurística simples)
            sorted_sentences = sorted(sentences, key=len, reverse=True)
            
            # Pegar as 2-3 frases mais longas de cada chunk
            frases_importantes.extend(sorted_sentences[:min(3, len(sorted_sentences))])
        
        # Limitar o tamanho total do resumo
        resumo = ". ".join(frases_importantes)
        if len(resumo) > max_length:
            # Tenta cortar em um ponto final para preservar frases completas
            ultima_frase = resumo[:max_length].rfind('.')
            if ultima_frase > 0:
                resumo = resumo[:ultima_frase + 1]
            else:
                resumo = resumo[:max_length] + "..."
        
        return resumo
    except Exception as e:
        print(f"Erro ao gerar resumo: {e}")
        return texto[:max_length] + "..."  # Fallback para corte simples
