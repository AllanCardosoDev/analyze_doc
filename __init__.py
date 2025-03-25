"""
Analyse Doc - Plataforma de análise de documentos com IA
--------------------------------------------------------

Um sistema completo para processamento, análise e visualização de documentos,
utilizando modelos avançados de IA e processamento de linguagem natural.

Módulos:
- loaders: Carregamento e pré-processamento de documentos
- processors: Análise avançada e extração de informações
- visualizers: Geração de visualizações e gráficos
- security: Segurança e privacidade de dados
- collaborative: Colaboração e compartilhamento
"""

__version__ = "1.0.0"
__author__ = "Equipe Analyse Doc"
__email__ = "contato@analysedoc.com"

import os
import logging
from datetime import datetime

# Configuração de logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{log_dir}/analyse_doc_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

# Verificações iniciais de dependências
try:
    import streamlit
    import langchain
    import pandas
except ImportError as e:
    logging.error(f"Erro ao importar dependências: {e}")
    logging.error("Execute 'pip install -r requirements.txt' para instalar todas as dependências.")
    raise

# Verificação de chaves de API
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    logging.warning("Nenhuma chave de API encontrada. Configure as chaves em .env para utilizar modelos de IA.")

# Verificação de OCR
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    logging.warning("OCR não disponível. Instale 'pytesseract' e 'Pillow' para habilitar o OCR.")

# Verificação de Spacy
try:
    import spacy
    SPACY_AVAILABLE = True
    # Verificar modelos instalados
    try:
        spacy.load("pt_core_news_sm")
    except OSError:
        logging.warning("Modelo Spacy para português não encontrado. Execute 'python -m spacy download pt_core_news_sm'")
except ImportError:
    SPACY_AVAILABLE = False
    logging.warning("Spacy não disponível. Instale 'spacy' para habilitar a extração de entidades.")

# Exporta as funções e classes principais
from loaders import (
    carrega_site, carrega_youtube, carrega_pdf, carrega_docx, 
    carrega_csv, carrega_txt, carrega_xlsx, carrega_pptx,
    carrega_json, carrega_markdown, carrega_html, carrega_imagem
)

from processors import (
    gera_resumo, extrai_entidades, analisa_sentimento,
    detecta_topicos, classifica_documento, extrai_tabelas,
    extrai_dados_financeiros, verifica_fatos, detecta_vies,
    extrai_formulas, extrai_referencias
)

from visualizers import (
    gera_nuvem_palavras, gera_grafico_entidades, gera_mapa_conexoes,
    gera_timeline, gera_grafico_estatisticas, gera_heatmap
)

from security import (
    anonimizar_texto, criptografa_documento, descriptografa_documento,
    hash_senha, verifica_senha, gera_chave_api
)

from collaborative import (
    cria_comentario, compartilha_documento, cria_versao,
    compara_versoes, cria_tarefa, gera_link_compartilhamento
)

# Informações do sistema
logging.info(f"Analyse Doc v{__version__} inicializado")
logging.info(f"OCR disponível: {OCR_AVAILABLE}")
logging.info(f"Spacy disponível: {SPACY_AVAILABLE}")
