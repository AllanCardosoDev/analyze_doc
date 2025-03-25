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
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument

def carrega_site(url):
    """Carrega texto de um site usando WebBaseLoader."""
    if not url:
        return "⚠️ URL não fornecida."
        
    documento = ""
    for i in range(5):  # Tenta 5 vezes
        try:
            # Usar um user-agent aleatório para evitar bloqueios
            os.environ["USER_AGENT"] = UserAgent().random
            
            # Verificar formato da URL
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
                
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = "\n\n".join([doc.page_content for doc in lista_documentos])
            if documento:
                break
        except Exception as e:
            print(f"Tentativa {i+1} falhou: {e}")
            sleep(3)  # Aguarda 3 segundos antes de tentar novamente
            
    if not documento:
        return "⚠️ Não foi possível carregar o site após múltiplas tentativas."
        
    return documento

def carrega_youtube(video_id, proxy=None):
    """
    Carrega legendas de vídeos do YouTube com suporte a proxy.
    
    Args:
        video_id: ID ou URL do vídeo do YouTube
        proxy: String de proxy no formato 'http://usuário:senha@host:porta'
    """
    if not video_id:
        return "⚠️ Nenhum ID ou URL de vídeo fornecido."
        
    try:
        # Extrai o video_id de uma URL completa, se for fornecida
        if "youtube.com" in video_id or "youtu.be" in video_id:
            if "youtube.com/watch?v=" in video_id:
                video_id = video_id.split("youtube.com/watch?v=")[1].split("&")[0]
            elif "youtu.be/" in video_id:
                video_id = video_id.split("youtu.be/")[1].split("?")[0]
        
        # Importa diretamente o youtube_transcript_api para mais controle
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api.formatters import TextFormatter
        
        # Configuração de proxy, se fornecido
        proxies = None
        if proxy:
            proxies = {'http': proxy, 'https': proxy}
        
        # Tenta primeiro com o idioma português, depois com inglês, depois com auto-geradas
        languages = ['pt', 'pt-BR', 'en']
        
        try:
            # Obtém as legendas diretamente, com suporte a proxy
            if proxies:
                from youtube_transcript_api import TranscriptListFetcher
                
                fetcher = TranscriptListFetcher(proxies=proxies)
                transcript_list = fetcher.fetch_transcript_list(video_id)
                transcript = transcript_list.find_transcript(languages)
                transcripts = transcript.fetch()
            else:
                # Método padrão sem proxy
                transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        except Exception as e:
            # Se falhar com os idiomas específicos, tenta com legendas auto-geradas
            if "No transcript" in str(e):
                try:
                    transcripts = YouTubeTranscriptApi.get_transcript(video_id)
                except Exception as inner_e:
                    return f"❌ Erro ao obter legendas: {inner_e}"
            else:
                raise e
        
        # Formata as legendas em um texto contínuo
        if not transcripts:
            return "⚠️ Não foi possível encontrar legendas para este vídeo."
        
        # Combina os segmentos de legendas em um texto formatado
        formatter = TextFormatter()
        texto_formatado = formatter.format_transcript(transcripts)
        
        if not texto_formatado:
            # Fallback para método manual se o formatter falhar
            texto_completo = ""
            for entry in transcripts:
                texto_completo += f"{entry.get('text', '')} "
            return texto_completo
            
        return texto_formatado
        
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
    """Carrega dados de arquivos CSV."""
    try:
        loader = CSVLoader(caminho)
        lista_documentos = loader.load()
        if not lista_documentos:
            return "⚠️ O arquivo CSV está vazio ou não contém dados válidos."
            
        return "\n\n".join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        return f"❌ Erro ao carregar CSV: {e}"

def carrega_pdf(caminho):
    """Carrega e extrai texto de um PDF."""
    try:
        loader = PyPDFLoader(caminho)
        documentos = loader.load()
        
        if not documentos:
            return "⚠️ O PDF está vazio ou não foi possível extrair conteúdo."
            
        texto = "\n\n".join([doc.page_content for doc in documentos])
        return texto if texto.strip() else "⚠️ O PDF não contém texto extraível. Pode ser um PDF digitalizado que requer OCR."
    except Exception as e:
        return f"❌ Erro ao carregar PDF: {e}"

def carrega_txt(caminho):
    """Carrega e extrai texto de um arquivo TXT."""
    try:
        loader = TextLoader(caminho)
        lista_documentos = loader.load()
        
        if not lista_documentos:
            return "⚠️ O arquivo TXT está vazio."
            
        return "\n\n".join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        return f"❌ Erro ao carregar TXT: {e}"

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
        
        # Verificar se temos conteúdo
        if not texto_completo:
            return "⚠️ O documento DOCX está vazio ou não contém texto."
            
        resultado = "\n\n".join(texto_completo)
        return resultado
    except Exception as e:
        return f"❌ Erro ao carregar DOCX: {e}"

def gera_resumo(texto, max_length=1000):
    """
    Gera um resumo automático do texto usando chunking e extração de frases importantes.
    Esta é uma implementação básica. Para resultados melhores, use os modelos de LLM.
    """
    if not texto or len(texto) < 200:
        return texto  # Texto muito curto, não precisa resumir
        
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
            # Filtrar frases muito curtas
            sentences = [s for s in sentences if len(s) > 30]
            # Selecionar as frases mais longas (geralmente mais informativas)
            sorted_sentences = sorted(sentences, key=len, reverse=True)
            # Pegar as 2-3 frases mais longas de cada chunk
            frases_importantes.extend(sorted_sentences[:min(3, len(sorted_sentences))])
        
        # Limitar o tamanho total do resumo
        resumo = ". ".join(frases_importantes)
        if len(resumo) > max_length:
            # Cortar no ponto final mais próximo para evitar frases incompletas
            idx = resumo[:max_length].rfind('.')
            if idx > 0:
                resumo = resumo[:idx + 1]  # Inclui o ponto final
            else:
                resumo = resumo[:max_length] + "..."
        
        return resumo
    except Exception as e:
        print(f"Erro ao gerar resumo: {e}")
        return "Não foi possível gerar um resumo automático. Usando documento original."

def traduz_texto(texto, idioma_destino='pt'):
    """
    Traduz o texto para o idioma especificado.
    Requer que um modelo de LLM esteja disponível.
    """
    try:
        from langchain_community.document_transformers import DoctranTextTranslator
        
        # Criar um documento Langchain
        doc = LangchainDocument(page_content=texto)
        
        # Configurar o tradutor
        translator = DoctranTextTranslator(
            language=idioma_destino
        )
        
        # Traduzir o documento
        documentos_traduzidos = translator.transform_documents([doc])
        
        return documentos_traduzidos[0].page_content
    except ImportError:
        print("DoctranTextTranslator não está disponível. Voltando ao texto original.")
        return texto
    except Exception as e:
        print(f"Erro ao traduzir texto: {e}")
        return f"Não foi possível traduzir o texto. Usando original. Erro: {e}"
