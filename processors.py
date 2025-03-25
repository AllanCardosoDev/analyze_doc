"""
Módulo de processadores avançados para o Analyse Doc.
Contém funções especializadas para verificação de fatos, detecção de viés,
extração de fórmulas e referências bibliográficas.
"""

import re
import json
import pandas as pd
import numpy as np
from collections import Counter
import requests
from bs4 import BeautifulSoup
import streamlit as st

# Funções importadas para compatibilidade com app.py
from loaders import (
    extrai_entidades, 
    analisa_sentimento, 
    detecta_topicos, 
    classifica_documento, 
    extrai_tabelas, 
    extrai_dados_financeiros
)

def verifica_fatos(texto, nivel="básico"):
    """
    Verifica afirmações factuais no texto.
    
    Args:
        texto: O texto a ser verificado
        nivel: Nível de verificação (básico, intermediário, avançado)
        
    Returns:
        Dicionário com afirmações verificadas
    """
    # Esta é uma implementação simulada
    # Uma implementação real exigiria uma API de fact-checking ou LLM avançado
    
    # Extrai afirmações que parecem factuais
    afirmacoes = []
    
    # Identifica frases com indicadores de afirmações
    indicadores = ["é", "são", "foi", "foram", "tem", "têm", "possui", "contém", 
                   "afirma", "afirmou", "segundo", "de acordo", "conforme", 
                   "estudo", "pesquisa", "dados"]
    
    sentencas = re.split(r'(?<=[.!?])\s+', texto)
    
    for sentenca in sentencas:
        if any(ind in sentenca.lower() for ind in indicadores):
            if len(sentenca.split()) > 5:  # Ignora sentenças muito curtas
                afirmacoes.append(sentenca)
    
    # Limita o número de afirmações analisadas
    afirmacoes = afirmacoes[:10]
    
    # Simulação de verificação
    resultado = []
    
    for afirmacao in afirmacoes:
        # Simulação simples de verificação
        confianca = np.random.choice(
            ["Alta", "Média", "Baixa", "Inconclusiva"],
            p=[0.3, 0.3, 0.2, 0.2]
        )
        
        status = np.random.choice(
            ["Verificado", "Parcialmente verificado", "Não verificado", "Falso"],
            p=[0.4, 0.3, 0.2, 0.1]
        )
        
        resultado.append({
            "afirmacao": afirmacao,
            "status": status,
            "confianca": confianca,
            "fonte": "Análise automática"
        })
    
    return resultado

def detecta_vies(texto):
    """
    Detecta viés linguístico no texto (político, de gênero, etc.)
    
    Args:
        texto: O texto a ser analisado
        
    Returns:
        Dicionário com análise de viés
    """
    # Esta é uma implementação simulada
    # Uma implementação real usaria modelos específicos para detecção de viés
    
    texto_lower = texto.lower()
    
    # Dicionários de palavras indicativas de viés (simplificado)
    vies_politico = {
        "esquerda": ["socialista", "comunista", "progressista", "petista", "welfare", "sindical"],
        "direita": ["liberal", "conservador", "privatização", "livre mercado", "bolsonarista"]
    }
    
    vies_genero = {
        "masculino": ["homem", "ele", "dele", "senhor", "pai", "irmão", "tio"],
        "feminino": ["mulher", "ela", "dela", "senhora", "mãe", "irmã", "tia"]
    }
    
    # Conta ocorrências
    contagem = {
        "político": {
            "esquerda": sum(texto_lower.count(palavra) for palavra in vies_politico["esquerda"]),
            "direita": sum(texto_lower.count(palavra) for palavra in vies_politico["direita"])
        },
        "gênero": {
            "masculino": sum(texto_lower.count(palavra) for palavra in vies_genero["masculino"]),
            "feminino": sum(texto_lower.count(palavra) for palavra in vies_genero["feminino"])
        }
    }
    
    # Analisa proporções e determina viés
    resultado = {
        "político": "neutro",
        "gênero": "equilibrado",
        "nível_geral": "baixo"
    }
    
    # Viés político
    total_politico = contagem["político"]["esquerda"] + contagem["político"]["direita"]
    if total_politico > 0:
        prop_esquerda = contagem["político"]["esquerda"] / total_politico
        
        if prop_esquerda > 0.7:
            resultado["político"] = "tendência à esquerda"
        elif prop_esquerda < 0.3:
            resultado["político"] = "tendência à direita"
    
    # Viés de gênero
    total_genero = contagem["gênero"]["masculino"] + contagem["gênero"]["feminino"]
    if total_genero > 0:
        prop_masculino = contagem["gênero"]["masculino"] / total_genero
        
        if prop_masculino > 0.7:
            resultado["gênero"] = "predominantemente masculino"
        elif prop_masculino < 0.3:
            resultado["gênero"] = "predominantemente feminino"
    
    # Nível geral baseado na força dos vieses detectados
    if resultado["político"] != "neutro" and resultado["gênero"] != "equilibrado":
        resultado["nível_geral"] = "significativo"
    elif resultado["político"] != "neutro" or resultado["gênero"] != "equilibrado":
        resultado["nível_geral"] = "moderado"
    
    return resultado

def extrai_formulas(texto):
    """
    Extrai fórmulas matemáticas e científicas do texto.
    
    Args:
        texto: O texto a ser analisado
        
    Returns:
        Lista de fórmulas encontradas
    """
    formulas = []
    
    # Padrões para fórmulas matemáticas simples
    # Nota: uma implementação completa usaria bibliotecas especializadas ou IA
    
    # Procura por expressões entre $ ou $$ (notação LaTeX)
    padrao_latex = r'\$\$(.*?)\$\$|\$(.*?)\$'
    matches_latex = re.findall(padrao_latex, texto, re.DOTALL)
    for match in matches_latex:
        formula = match[0] if match[0] else match[1]
        if formula.strip():
            formulas.append({"tipo": "latex", "formula": formula.strip()})
    
    # Procura por equações químicas
    padrao_quimico = r'([A-Z][a-z]?\d*)+\s*(?:[-+→⟶]\s*([A-Z][a-z]?\d*)+)+\s*'
    matches_quimicos = re.findall(padrao_quimico, texto)
    for match in matches_quimicos:
        formulas.append({"tipo": "química", "formula": ''.join(match)})
    
    # Procura por expressões matemáticas comuns
    padrao_math = r'[A-Za-z]+\s*=\s*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?(?:\s*[-+*/]\s*[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)*'
    matches_math = re.findall(padrao_math, texto)
    for match in matches_math:
        if '=' in match and len(match) > 3:  # Filtra fórmulas triviais
            formulas.append({"tipo": "matemática", "formula": match.strip()})
    
    # Remove duplicatas
    formulas_unicas = []
    formulas_vistas = set()
    
    for formula in formulas:
        if formula["formula"] not in formulas_vistas:
            formulas_vistas.add(formula["formula"])
            formulas_unicas.append(formula)
    
    return formulas_unicas

def extrai_referencias(texto):
    """
    Extrai referências bibliográficas do texto.
    
    Args:
        texto: O texto a ser analisado
        
    Returns:
        Lista de referências encontradas
    """
    referencias = []
    
    # Padrões para referências bibliográficas
    # Formato ABNT
    padrao_abnt = r'([A-Z][A-ZÀ-Ú]*,\s*[A-Z][a-zà-ú]*(?:\s+[A-Z][a-zà-ú]*)*)\.(?:.*?)(?:\d{4})'
    matches_abnt = re.findall(padrao_abnt, texto)
    for match in matches_abnt:
        if len(match) > 10:  # Filtra correspondências muito curtas
            referencias.append({"tipo": "ABNT", "texto": match.strip()})
    
    # Formato APA 
    padrao_apa = r'([A-Z][a-z]+,\s*[A-Z]\.(?:\s+[A-Z]\.)*)(?:.*?)(?:\(\d{4}\))'
    matches_apa = re.findall(padrao_apa, texto)
    for match in matches_apa:
        if len(match) > 5:
            referencias.append({"tipo": "APA", "texto": match.strip()})
    
    # Referências a URLs
    padrao_url = r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
    matches_url = re.findall(padrao_url, texto)
    for match in matches_url:
        referencias.append({"tipo": "URL", "texto": match})
    
    # Referências a DOI
    padrao_doi = r'(?:doi:|https?://doi.org/)?(10\.\d{4,9}/[-._;()/:A-Z0-9]+)'
    matches_doi = re.findall(padrao_doi, texto, re.IGNORECASE)
    for match in matches_doi:
        referencias.append({"tipo": "DOI", "texto": match})
    
    # Remove duplicatas
    referencias_unicas = []
    refs_vistas = set()
    
    for ref in referencias:
        if ref["texto"] not in refs_vistas:
            refs_vistas.add(ref["texto"])
            referencias_unicas.append(ref)
    
    return referencias_unicas[:20]  # Limitamos a 20 referências para não sobrecarregar

def valida_informacoes(texto, tipo_documento=None):
    """
    Valida informações específicas com base no tipo de documento.
    
    Args:
        texto: O texto a ser validado
        tipo_documento: O tipo de documento (opcional)
        
    Returns:
        Dicionário com resultados da validação
    """
    resultados = {
        "erros": [],
        "avisos": [],
        "inconsistencias": []
    }
    
    # Se o tipo não for especificado, tenta determinar
    if not tipo_documento:
        tipo_documento = classifica_documento(texto)
    
    # Validações específicas por tipo de documento
    if "contrato" in tipo_documento.lower():
        # Validação de contratos
        if not re.search(r'cláusula', texto, re.IGNORECASE):
            resultados["avisos"].append("Documento não contém cláusulas explícitas")
        
        # Verificar se há datas válidas
        datas = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', texto)
        if not datas:
            resultados["avisos"].append("Nenhuma data encontrada no contrato")
        
        # Verificar se há valores monetários
        valores = re.findall(r'R\$\s*\d+(?:[.,]\d+)*', texto)
        if not valores:
            resultados["avisos"].append("Nenhum valor monetário encontrado")
    
    elif "científico" in tipo_documento.lower():
        # Validação de artigos científicos
        if not re.search(r'abstract|resumo', texto, re.IGNORECASE):
            resultados["avisos"].append("Documento não contém resumo/abstract")
        
        if not re.search(r'referências|bibliografia|references', texto, re.IGNORECASE):
            resultados["avisos"].append("Documento não contém seção de referências")
        
        if not re.search(r'metodologia|método|methods', texto, re.IGNORECASE):
            resultados["avisos"].append("Documento não contém seção de metodologia")
    
    elif "email" in tipo_documento.lower():
        # Validação de emails
        if not re.search(r'prezado|caro|olá|bom dia|boa tarde', texto, re.IGNORECASE):
            resultados["avisos"].append("Email não contém saudação inicial")
        
        if not re.search(r'atenciosamente|cordialmente|abraços|att\.', texto, re.IGNORECASE):
            resultados["avisos"].append("Email não contém despedida")
    
    # Validações gerais para todos os tipos
    urls = re.findall(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b', texto)
    for url in urls:
        if url.endswith('.') or url.endswith(','):
            resultados["inconsistencias"].append(f"URL possivelmente incompleta: {url}")
    
    # Verifica inconsistências em datas
    datas = re.findall(r'\d{1,2}/\d{1,2}/\d{4}', texto)
    for data in datas:
        partes = data.split('/')
        if len(partes) == 3:
            dia, mes, ano = map(int, partes)
            if mes > 12 or dia > 31:
                resultados["erros"].append(f"Data inválida: {data}")
    
    return resultados
