"""
Módulo de carregadores para diferentes tipos de documentos.
Fornece funções para extrair texto de arquivos e conteúdo web.
"""

import os
import re
import streamlit as st
from langchain_community.document_loaders import (
    WebBaseLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

# Configurações e variáveis globais
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0'
]

def get_random_user_agent():
    """Retorna um User-Agent aleatório da lista."""
    import random
    return random.choice(USER_AGENTS)

def carrega_site(url):
    """
    Carrega texto de um site usando WebBaseLoader.
    
    Args:
        url: URL do site a ser carregado
        
    Returns:
        String com o conteúdo do site ou mensagem de erro
    """
    documento = ""
    
    try:
        # Configura o User-Agent
        os.environ["USER_AGENT"] = get_random_user_agent()
        
        # Verifica se a URL é válida
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Carrega o site com tratamento de erros
        loader = WebBaseLoader(url)
        lista_documentos = loader.load()
        
        # Concatena o conteúdo de todas as páginas
        documento = "\n\n".join([doc.page_content for doc in lista_documentos])
        
        # Verifica se conseguiu extrair conteúdo
        if not documento.strip():
            return "⚠️ Não foi possível extrair conteúdo do site. A página pode estar bloqueando acesso automatizado."
            
        return documento
    
    except Exception as e:
        return f"❌ Erro ao carregar o site: {str(e)}"

def carrega_youtube(video_id, proxy=None):
    """
    Carrega legendas de vídeos do YouTube.
    
    Args:
        video_id: ID ou URL do vídeo do YouTube
        proxy: String de proxy no formato 'http://host:porta'
        
    Returns:
        String com as legendas do vídeo ou mensagem de erro
    """
    try:
        # Importa a biblioteca de forma dinâmica para evitar erros no deploy
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            return "❌ Biblioteca youtube_transcript_api não está instalada."
        
        # Extrai o video_id de uma URL completa, se for fornecida
        if "youtube.com" in video_id or "youtu.be" in video_id:
            if "youtube.com/watch?v=" in video_id:
                video_id = video_id.split("youtube.com/watch?v=")[1].split("&")[0]
            elif "youtu.be/" in video_id:
                video_id = video_id.split("youtu.be/")[1].split("?")[0]
        
        # Tenta obter as legendas em diferentes idiomas
        languages = ['pt', 'pt-BR', 'en']
        
        # Configuração de proxy
        proxies = None
        if proxy:
            proxies = {'http': proxy, 'https': proxy}
        
        # Tenta obter transcrição
        try:
            if proxies:
                from youtube_transcript_api import TranscriptListFetcher
                fetcher = TranscriptListFetcher(proxies=proxies)
                transcript_list = fetcher.fetch_transcript_list(video_id)
                transcript = transcript_list.find_transcript(languages)
                transcripts = transcript.fetch()
            else:
                transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        except Exception as e:
            return f"""❌ Erro ao buscar legendas: {str(e)}
            
Possíveis soluções:
1. Verifique se o vídeo possui legendas nos idiomas disponíveis (português ou inglês)
2. Configure um proxy nas configurações adicionais
3. O YouTube pode estar bloqueando requisições da aplicação
            """
        
        # Verifica se encontrou legendas
        if not transcripts:
            return "⚠️ Não foram encontradas legendas para este vídeo."
        
        # Formata as legendas em texto
        texto_formatado = ""
        for entry in transcripts:
            # Adiciona timestamp se disponível
            if 'start' in entry:
                seconds = int(entry['start'])
                minutes = seconds // 60
                seconds %= 60
                texto_formatado += f"[{minutes:02d}:{seconds:02d}] "
            
            # Adiciona o texto da legenda
            texto_formatado += f"{entry.get('text', '')} "
        
        return texto_formatado
    
    except Exception as e:
        return f"❌ Erro ao processar vídeo do YouTube: {str(e)}"

def carrega_pdf(caminho):
    """
    Carrega e extrai texto de um PDF.
    
    Args:
        caminho: Caminho para o arquivo PDF
        
    Returns:
        String com o conteúdo do PDF
    """
    try:
        # Carrega o PDF usando PyPDFLoader do Langchain
        loader = PyPDFLoader(caminho)
        documentos = loader.load()
        
        # Concatena o conteúdo de todas as páginas
        texto = "\n\n".join([f"[Página {i+1}]\n{doc.page_content}" 
                            for i, doc in enumerate(documentos)])
        
        # Verifica se conseguiu extrair texto
        if not texto.strip():
            return "⚠️ O PDF não contém texto extraível. Pode ser um PDF escaneado que requer OCR."
        
        return texto
    
    except Exception as e:
        return f"❌ Erro ao processar PDF: {str(e)}"

def carrega_txt(caminho):
    """
    Carrega e extrai texto de um arquivo TXT.
    
    Args:
        caminho: Caminho para o arquivo TXT
        
    Returns:
        String com o conteúdo do arquivo TXT
    """
    try:
        # Tenta diferentes codificações em caso de erro
        encodings = ['utf-8', 'latin-1', 'cp1252']
        
        for encoding in encodings:
            try:
                with open(caminho, 'r', encoding=encoding) as file:
                    return file.read()
            except UnicodeDecodeError:
                continue
        
        # Se nenhuma codificação funcionou, tenta o loader do Langchain
        loader = TextLoader(caminho)
        documentos = loader.load()
        return "\n\n".join([doc.page_content for doc in documentos])
    
    except Exception as e:
        return f"❌ Erro ao carregar arquivo TXT: {str(e)}"

def carrega_csv(caminho):
    """
    Carrega dados de arquivos CSV.
    
    Args:
        caminho: Caminho para o arquivo CSV
        
    Returns:
        String com o conteúdo do CSV formatado
    """
    try:
        # Tenta usar o loader do Langchain
        loader = CSVLoader(caminho)
        documentos = loader.load()
        
        # Formata o conteúdo para melhor legibilidade
        conteudo = "\n\n".join([doc.page_content for doc in documentos])
        
        return conteudo
    
    except Exception as e:
        # Fallback: tenta carregar manualmente com pandas
        try:
            import pandas as pd
            df = pd.read_csv(caminho)
            return df.to_string()
        except:
            return f"❌ Erro ao carregar arquivo CSV: {str(e)}"

def carrega_docx(caminho):
    """
    Carrega e extrai texto de um arquivo DOCX.
    
    Args:
        caminho: Caminho para o arquivo DOCX
        
    Returns:
        String com o conteúdo do documento DOCX
    """
    try:
        # Importa python-docx se disponível
        try:
            from docx import Document
        except ImportError:
            return "❌ Biblioteca python-docx não está instalada."
        
        # Carrega o documento
        doc = Document(caminho)
        texto_completo = []
        
        # Extrai texto de parágrafos
        for para in doc.paragraphs:
            if para.text.strip():
                texto_completo.append(para.text)
        
        # Extrai texto de tabelas
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    texto_completo.append(" | ".join(row_text))
        
        # Junta todo o texto
        resultado = "\n\n".join(texto_completo)
        
        # Verifica se conseguiu extrair texto
        if not resultado.strip():
            return "⚠️ O documento DOCX não contém texto extraível."
        
        return resultado
    
    except Exception as e:
        return f"❌ Erro ao carregar DOCX: {str(e)}"

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
            sentences = [s + '.' for s in sentences if len(s.split()) > 5]
            
            # Ordenar frases por comprimento (uma heurística simples)
            sorted_sentences = sorted(sentences, key=len, reverse=True)
            
            # Pegar as 2-3 frases mais longas de cada chunk
            frases_importantes.extend(sorted_sentences[:min(3, len(sorted_sentences))])
        
        # Limitar o tamanho total do resumo
        resumo = " ".join(frases_importantes)
        if len(resumo) > max_length:
            # Tenta cortar em um ponto final para preservar frases completas
            ultima_frase = resumo[:max_length].rfind('.')
            if ultima_frase > 0:
                resumo = resumo[:ultima_frase + 1]
            else:
                resumo = resumo[:max_length] + "..."
        
        return resumo
    
    except Exception as e:
        st.warning(f"Erro ao gerar resumo: {str(e)}")
        # Fallback: corte simples do texto original
        return texto[:max_length] + "..."
