"""
Módulo de resumo automático de documentos para o Analyse Doc.
Este módulo implementa funcionalidades para gerar resumos detalhados de documentos.
"""

import re
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from fpdf import FPDF
import io
from loaders import gera_resumo, traduz_texto

# Baixar recursos NLTK necessários (executado apenas uma vez)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

def gerar_resumo_documento(documento, tipo_documento, config):
    """
    Gera um resumo detalhado do documento com base nas configurações definidas.
    
    Args:
        documento (str): Conteúdo do documento
        tipo_documento (str): Tipo do documento (PDF, DOCX, etc.)
        config (dict): Configurações de resumo
        
    Returns:
        dict: Resumo e metadados
    """
    # Extrair configurações
    max_length = config.get("max_length", 1500)
    idioma = config.get("idioma", "pt")
    usar_llm = config.get("usar_llm", True)
    
    # Gerar resumo baseado nas configurações
    if usar_llm and "llm_chain" in config:
        # Usar modelo LLM para resumo
        prompt_resumo = f"""
        Crie um resumo conciso e informativo do seguinte documento. 
        O resumo deve ter no máximo {max_length} caracteres e capturar 
        os pontos principais, ideias centrais e informações mais relevantes.
        
        DOCUMENTO:
        {documento[:20000]}  # Limitando para não exceder tokens
        
        RESUMO:
        """
        
        resumo = config["llm_chain"].invoke(prompt_resumo).content
    else:
        # Usar método de extração de frases importantes
        resumo = gera_resumo(documento, max_length)
    
    # Traduzir se necessário
    if idioma != "pt" and config.get("tradutor_disponivel", False):
        resumo = traduz_texto(resumo, idioma)
    
    # Gerar seções adicionais do resumo
    seccoes = {}
    
    # Verificar se devemos incluir tópicos principais
    if config.get("incluir_topicos", True):
        if usar_llm and "llm_chain" in config:
            # Usar LLM para identificar tópicos
            prompt_topicos = f"""
            Identifique os 3-5 tópicos principais abordados no seguinte documento.
            Para cada tópico, forneça um título curto e conciso.
            
            DOCUMENTO:
            {documento[:15000]}
            
            TÓPICOS PRINCIPAIS:
            """
            
            topicos = config["llm_chain"].invoke(prompt_topicos).content
            seccoes["Tópicos Principais"] = topicos
        else:
            # Usar método de extração de tópicos
            topicos = extrair_topicos(documento)
            seccoes["Tópicos Principais"] = topicos
    
    # Verificar se devemos incluir termos-chave
    if config.get("incluir_termos", True):
        termos = extrair_termos_chave(documento, idioma)
        seccoes["Termos-Chave"] = termos
    
    # Verificar se devemos analisar estrutura
    if config.get("analisar_estrutura", False) and usar_llm and "llm_chain" in config:
        prompt_estrutura = f"""
        Analise a estrutura do seguinte documento e identifique suas principais seções.
        Forneça um breve resumo da organização do documento.
        
        DOCUMENTO:
        {documento[:15000]}
        
        ANÁLISE DA ESTRUTURA:
        """
        
        estrutura = config["llm_chain"].invoke(prompt_estrutura).content
        seccoes["Estrutura do Documento"] = estrutura
    
    # Montar metadados
    metadados = {
        "tipo": tipo_documento,
        "tamanho_original": len(documento),
        "tamanho_resumo": len(resumo),
        "data_processamento": datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    }
    
    # Gerar PDF do resumo com todas as seções
    pdf_bytes = gera_pdf_resumo_completo(resumo, seccoes, tipo_documento, metadados)
    
    return {
        "resumo": resumo,
        "seccoes": seccoes,
        "metadados": metadados,
        "pdf_bytes": pdf_bytes
    }

def extrair_topicos(texto):
    """
    Extrai tópicos principais do texto usando análise simples de frequência.
    
    Args:
        texto (str): Texto do documento
        
    Returns:
        str: Texto formatado com os tópicos principais
    """
    # Dividir o texto em parágrafos
    paragrafos = texto.split('\n\n')
    
    # Selecionar os parágrafos mais informativos (primeiros e últimos tendem a ser mais relevantes)
    paragrafos_selecionados = paragrafos[:2]
    if len(paragrafos) > 4:
        paragrafos_selecionados.extend(paragrafos[-2:])
    
    # Extrair as primeiras frases de cada parágrafo como "tópico"
    topicos = []
    for paragrafo in paragrafos_selecionados:
        if paragrafo.strip():
            frases = sent_tokenize(paragrafo)
            if frases:
                # Limitar a frase a 100 caracteres para ser concisa
                frase = frases[0][:100]
                if len(frases[0]) > 100:
                    frase += "..."
                topicos.append(f"• {frase}")
    
    # Filtrar tópicos para evitar repetições
    topicos_unicos = []
    for topico in topicos:
        # Verificar se é muito similar a algum tópico já adicionado
        if not any(similaridade_texto(topico, t) > 0.7 for t in topicos_unicos):
            topicos_unicos.append(topico)
    
    return "\n".join(topicos_unicos)

def extrair_termos_chave(texto, idioma='pt'):
    """
    Extrai os termos mais relevantes do texto.
    
    Args:
        texto (str): Texto do documento
        idioma (str): Código do idioma para stopwords
        
    Returns:
        str: Texto formatado com os termos-chave
    """
    # Verificar idioma para stopwords
    idioma_nltk = idioma
    if idioma == 'pt':
        idioma_nltk = 'portuguese'
    elif idioma == 'en':
        idioma_nltk = 'english'
    elif idioma == 'es':
        idioma_nltk = 'spanish'
    elif idioma == 'fr':
        idioma_nltk = 'french'
    
    try:
        stop_words = set(stopwords.words(idioma_nltk))
    except:
        # Fallback para inglês se o idioma não estiver disponível
        stop_words = set(stopwords.words('english'))
    
    # Tokenizar palavras
    palavras = word_tokenize(texto.lower())
    
    # Filtrar stopwords e caracteres não alfabéticos
    palavras_filtradas = [w for w in palavras if w.isalpha() and w not in stop_words and len(w) > 3]
    
    # Calcular frequência
    fdist = FreqDist(palavras_filtradas)
    
    # Pegar os 15 termos mais frequentes
    termos_frequentes = fdist.most_common(15)
    
    # Formatar resultado
    resultado = []
    for termo, freq in termos_frequentes:
        resultado.append(f"• {termo.capitalize()} ({freq})")
    
    return "\n".join(resultado)

def similaridade_texto(texto1, texto2):
    """
    Calcula uma similaridade simples entre dois textos.
    
    Args:
        texto1 (str): Primeiro texto
        texto2 (str): Segundo texto
        
    Returns:
        float: Valor de similaridade entre 0 e 1
    """
    # Converter para minúsculas e tokenizar
    palavras1 = set(word_tokenize(texto1.lower()))
    palavras2 = set(word_tokenize(texto2.lower()))
    
    # Calcular interseção e união
    intersecao = palavras1.intersection(palavras2)
    uniao = palavras1.union(palavras2)
    
    # Retornar similaridade de Jaccard
    if not uniao:
        return 0
    return len(intersecao) / len(uniao)

def gera_pdf_resumo_completo(texto_resumo, seccoes, tipo_documento, metadados):
    """
    Gera um PDF detalhado com o resumo do documento e seções adicionais.
    
    Args:
        texto_resumo (str): Texto resumido principal
        seccoes (dict): Dicionário com seções adicionais
        tipo_documento (str): Tipo do documento original
        metadados (dict): Metadados do documento
        
    Returns:
        bytes: Conteúdo do PDF em bytes
    """
    class PDF(FPDF):
        def header(self):
            # Logo e cabeçalho
            self.set_font('Arial', 'B', 15)
            self.cell(0, 10, 'Analyse Doc - Resumo Detalhado', 0, 1, 'C')
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)
            
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Página {self.page_no()}/{{nb}}', 0, 0, 'C')
            
        def section_title(self, title):
            self.set_font('Arial', 'B', 12)
            self.set_fill_color(230, 230, 230)
            self.cell(0, 8, title, 0, 1, 'L', 1)
            self.ln(4)
            
        def section_content(self, content):
            self.set_font('Arial', '', 11)
            self.multi_cell(0, 6, content)
            self.ln(4)
    
    # Criar objeto PDF
    pdf = PDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    
    # Adicionar metadados
    pdf.set_author('Analyse Doc')
    pdf.set_title(f'Resumo Detalhado - {tipo_documento}')
    pdf.set_creator('Analyse Doc')
    
    # Informações do documento
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Informações do Documento:', 0, 1)
    
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 8, f'Tipo: {tipo_documento}', 0, 1)
    pdf.cell(0, 8, f'Data de Processamento: {metadados.get("data_processamento", "")}', 0, 1)
    pdf.cell(0, 8, f'Tamanho do Documento: {metadados.get("tamanho_original", 0)} caracteres', 0, 1)
    pdf.cell(0, 8, f'Tamanho do Resumo: {metadados.get("tamanho_resumo", 0)} caracteres', 0, 1)
    
    # Adicionar linha separadora
    pdf.line(10, pdf.get_y() + 5, 200, pdf.get_y() + 5)
    pdf.ln(10)
    
    # Resumo principal
    pdf.section_title('Resumo do Conteúdo')
    pdf.section_content(texto_resumo)
    
    # Adicionar seções adicionais
    for titulo, conteudo in seccoes.items():
        pdf.section_title(titulo)
        pdf.section_content(conteudo)
    
    # Adicionar aviso sobre resumo automático
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.multi_cell(0, 5, 'Nota: Este resumo foi gerado automaticamente pela Analyse Doc e pode não representar completamente o conteúdo original do documento.')
    
    # Retornar o PDF em bytes
    pdf_output = io.BytesIO()
    pdf.output(pdf_output)
    return pdf_output.getvalue()
