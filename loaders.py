import os
from time import sleep
import streamlit as st
from langchain_community.document_loaders import (
    WebBaseLoader,
    CSVLoader,
    PyPDFLoader,
    TextLoader
)
from fake_useragent import UserAgent
import unicodedata

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

def carrega_youtube(video_url):
    """Versão corrigida para carregar legendas do YouTube em português auto-geradas."""
    try:
        # Importar diretamente
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Limpeza e extração do ID do vídeo
        if "youtube.com" in video_url or "youtu.be" in video_url:
            if "v=" in video_url:
                video_id = video_url.split("v=")[1].split("&")[0]
            elif "youtu.be/" in video_url:
                video_id = video_url.split("youtu.be/")[1].split("?")[0]
            else:
                video_id = video_url
        else:
            video_id = video_url
            
        # Primeiro, tentar obter legendas em português (geradas automaticamente)
        try:
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt'])
        except:
            try:
                # Segunda tentativa: auto-geradas em português-Brasil
                transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt-BR'])
            except:
                try:
                    # Terceira tentativa: inglês
                    transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                except:
                    # Última tentativa: qualquer legenda disponível
                    available_transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
                    if available_transcripts:
                        # Usar a primeira legenda disponível
                        for transcript in available_transcripts:
                            transcript_data = transcript.fetch()
                            break
                    else:
                        return "⚠️ Nenhuma legenda disponível para este vídeo."
        
        # Processamento do resultado
        texto_legendas = ""
        for item in transcript_data:
            if isinstance(item, dict) and 'text' in item:
                texto_legendas += item['text'] + " "
        
        return texto_legendas if texto_legendas else "⚠️ Não foi possível extrair texto das legendas."
            
    except Exception as e:
        return f"❌ Erro ao carregar YouTube: {e}"

def limpa_texto(texto_bruto):
    """Função auxiliar para limpar texto com problemas de codificação."""
    # Normaliza o texto para remover caracteres problemáticos
    texto_normalizado = unicodedata.normalize('NFKD', texto_bruto)
    
    # Remove caracteres não-ASCII
    texto_limpo = ''
    for char in texto_normalizado:
        if ord(char) < 128 or char.isspace():
            texto_limpo += char
        else:
            # Substitui caracteres não-ASCII por espaço ou equivalente ASCII quando possível
            if char in 'áàãâäåæ':
                texto_limpo += 'a'
            elif char in 'éèêë':
                texto_limpo += 'e'
            elif char in 'íìîï':
                texto_limpo += 'i'
            elif char in 'óòõôöø':
                texto_limpo += 'o'
            elif char in 'úùûü':
                texto_limpo += 'u'
            elif char in 'ç':
                texto_limpo += 'c'
            elif char in 'ñ':
                texto_limpo += 'n'
            else:
                texto_limpo += ' '
    
    return texto_limpo

def carrega_csv(caminho):
    """Carrega dados de arquivos CSV."""
    try:
        loader = CSVLoader(caminho)
        lista_documentos = loader.load()
        texto = "\n\n".join([doc.page_content for doc in lista_documentos])
        return limpa_texto(texto)
    except Exception as e:
        return f"❌ Erro ao carregar CSV: {e}"

def carrega_pdf(caminho):
    """Carrega e extrai texto de um PDF com tratamento de codificação."""
    try:
        loader = PyPDFLoader(caminho)
        documentos = loader.load()
        texto = "\n\n".join([doc.page_content for doc in documentos])
        
        if not texto.strip():
            return "⚠️ O PDF não contém texto extraível."
        
        # Limpa o texto para evitar problemas de codificação
        return limpa_texto(texto)
    except Exception as e:
        return f"❌ Erro ao carregar PDF: {e}"

def carrega_txt(caminho):
    """Carrega e extrai texto de um arquivo TXT."""
    try:
        loader = TextLoader(caminho)
        lista_documentos = loader.load()
        texto = "\n\n".join([doc.page_content for doc in lista_documentos])
        return limpa_texto(texto)
    except Exception as e:
        return f"❌ Erro ao carregar TXT: {e}"
