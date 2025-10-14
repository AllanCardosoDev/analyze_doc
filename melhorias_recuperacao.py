"""
Sistema melhorado de recuperação de informações com contexto estrutural.
"""
import re
from typing import List, Dict, Optional, Tuple
from langchain.schema import Document
import streamlit as st
import logging

from diagnostico import DocumentDiagnostic

logger = logging.getLogger(__name__)


class SmartRetriever:
    """Recuperador inteligente que entende a estrutura do documento."""
    
    def __init__(self):
        self.diagnostico = DocumentDiagnostic()
        self.estrutura = None
        self.mapa_documento = None
    
    def initialize_with_document(self, documento: str) -> Dict:
        """
        Inicializa o recuperador com um documento.
        
        Args:
            documento: Conteúdo do documento
            
        Returns:
            dict: Informações da estrutura
        """
        logger.info("Inicializando SmartRetriever...")
        
        # Analisar estrutura
        self.estrutura = self.diagnostico.analizar_estrutura_documento(documento)
        
        # Criar mapa
        self.mapa_documento = self.diagnostico.criar_mapa_documento(
            documento, self.estrutura
        )
        
        # Salvar na sessão
        st.session_state['estrutura_documento'] = self.estrutura
        st.session_state['mapa_documento'] = self.mapa_documento
        
        return {
            'capitulos': len(self.estrutura['capitulos']),
            'paginas': len(self.estrutura['paginas']),
            'indice_encontrado': len(self.estrutura['indices']) > 0
        }
    
    def retrieve_with_structure(
        self, 
        query: str, 
        chunks: List[Document],
        k: int = 3
    ) -> Tuple[List[Document], str]:
        """
        Recupera chunks considerando a estrutura do documento.
        
        Args:
            query: Pergunta do usuário
            chunks: Lista de chunks
            k: Número de chunks a recuperar
            
        Returns:
            tuple: (chunks relevantes, contexto adicional)
        """
        # Detectar tipo de pergunta
        tipo_pergunta = self._detectar_tipo_pergunta(query)
        
        contexto_adicional = ""
        chunks_selecionados = []
        
        if tipo_pergunta == 'estrutura_geral':
            # Perguntas sobre estrutura geral (quantos capítulos, resumo geral)
            contexto_adicional = f"""
ESTRUTURA DO DOCUMENTO:
{self.mapa_documento}

Use essas informações para responder sobre a estrutura geral do documento.
"""
            # Pegar chunks do início e do índice
            chunks_selecionados = self._get_structural_chunks(chunks)
        
        elif tipo_pergunta == 'capitulo_especifico':
            # Perguntas sobre capítulos específicos
            numero_cap = self._extrair_numero_capitulo(query)
            
            if numero_cap and self.estrutura:
                # Tentar extrair o capítulo completo
                documento = st.session_state.get('documento_completo', '')
                conteudo_cap = self.diagnostico.extrair_capitulo_especifico(
                    documento, numero_cap, self.estrutura
                )
                
                if conteudo_cap and not conteudo_cap.startswith("Capítulo"):
                    contexto_adicional = f"""
CONTEÚDO COMPLETO DO CAPÍTULO {numero_cap}:
{conteudo_cap[:5000]}  # Limitar para não exceder tokens

Use ESTE conteúdo para responder sobre o capítulo {numero_cap}.
"""
                    # Buscar chunks relacionados também
                    chunks_selecionados = self._buscar_chunks_capitulo(
                        chunks, numero_cap, k
                    )
        
        elif tipo_pergunta == 'conteudo_especifico':
            # Perguntas sobre conteúdo específico
            # Usar busca normal melhorada
            chunks_selecionados = self._busca_inteligente(query, chunks, k * 2)
        
        else:
            # Busca padrão
            chunks_selecionados = self._busca_inteligente(query, chunks, k)
        
        # Se não encontrou chunks suficientes, adicionar mais
        if len(chunks_selecionados) < k:
            chunks_adicionais = self._busca_inteligente(
                query, chunks, k - len(chunks_selecionados)
            )
            chunks_selecionados.extend(chunks_adicionais)
        
        # Remover duplicatas
        chunks_unicos = []
        ids_vistos = set()
        for chunk in chunks_selecionados:
            chunk_id = chunk.metadata.get('chunk_id', id(chunk))
            if chunk_id not in ids_vistos:
                chunks_unicos.append(chunk)
                ids_vistos.add(chunk_id)
        
        return chunks_unicos[:k * 2], contexto_adicional  # Retornar mais chunks
    
    def _detectar_tipo_pergunta(self, query: str) -> str:
        """Detecta o tipo de pergunta do usuário."""
        query_lower = query.lower()
        
        # Estrutura geral
        if any(palavra in query_lower for palavra in [
            'quantos capítulos', 'quantos capitulos',
            'estrutura', 'organização', 'índice', 'sumário',
            'lista de capítulos', 'todos os capítulos'
        ]):
            return 'estrutura_geral'
        
        # Capítulo específico
        if any(palavra in query_lower for palavra in [
            'primeiro capítulo', 'segundo capítulo', 'terceiro capítulo',
            'último capítulo', 'capítulo', 'capitulo'
        ]) or re.search(r'cap[íi]tulo\s+\d+', query_lower):
            return 'capitulo_especifico'
        
        # Conteúdo específico
        return 'conteudo_especifico'
    
    def _extrair_numero_capitulo(self, query: str) -> Optional[int]:
        """Extrai o número do capítulo da query."""
        query_lower = query.lower()
        
        # Números escritos por extenso
        numeros_extenso = {
            'primeiro': 1, 'segunda': 2, 'terceiro': 3, 'quarto': 4,
            'quinto': 5, 'sexto': 6, 'sétimo': 7, 'oitavo': 8,
            'nono': 9, 'décimo': 10, 'último': -1, 'ultima': -1
        }
        
        for palavra, numero in numeros_extenso.items():
            if palavra in query_lower:
                if numero == -1 and self.estrutura:
                    # Retornar o último capítulo
                    if self.estrutura['capitulos']:
                        return int(self.estrutura['capitulos'][-1]['numero'])
                return numero
        
        # Números diretos
        match = re.search(r'cap[íi]tulo\s+(\d+)', query_lower)
        if match:
            return int(match.group(1))
        
        match = re.search(r'\b(\d+)[ºª°]?\s+cap[íi]tulo', query_lower)
        if match:
            return int(match.group(1))
        
        return None
    
    def _get_structural_chunks(self, chunks: List[Document]) -> List[Document]:
        """Pega chunks que contêm informação estrutural."""
        structural_chunks = []
        
        for chunk in chunks[:50]:  # Verificar os primeiros 50 chunks
            texto = chunk.page_content.lower()
            if any(palavra in texto for palavra in [
                'sumário', 'índice', 'capítulo', 'contents',
                'table of contents', 'prefácio'
            ]):
                structural_chunks.append(chunk)
        
        return structural_chunks[:5]
    
    def _buscar_chunks_capitulo(
        self, 
        chunks: List[Document], 
        numero_cap: int,
        k: int
    ) -> List[Document]:
        """Busca chunks que pertencem a um capítulo específico."""
        chunks_capitulo = []
        
        # Padrões para identificar o capítulo
        padroes = [
            f'cap[íi]tulo {numero_cap}',
            f'capítulo {numero_cap}',
            f'chapter {numero_cap}',
            f'^{numero_cap}[\\s.-]',
        ]
        
        for chunk in chunks:
            texto = chunk.page_content.lower()
            for padrao in padroes:
                if re.search(padrao, texto):
                    chunks_capitulo.append(chunk)
                    break
        
        return chunks_capitulo[:k]
    
    def _busca_inteligente(
        self, 
        query: str, 
        chunks: List[Document],
        k: int
    ) -> List[Document]:
        """Busca inteligente com múltiplos critérios."""
        from config import STOPWORDS_PT
        
        # Normalizar query
        query_norm = re.sub(r'[^\w\s]', '', query.lower())
        keywords = [
            word for word in query_norm.split() 
            if word not in STOPWORDS_PT and len(word) > 2
        ]
        
        # Expandir keywords com sinônimos comuns
        sinonimos = {
            'fala': ['fala', 'trata', 'aborda', 'discute', 'explica'],
            'conteúdo': ['conteúdo', 'assunto', 'tema', 'tópico'],
            'capítulo': ['capítulo', 'capitulo', 'seção', 'secao', 'parte']
        }
        
        keywords_expandidas = keywords.copy()
        for kw in keywords:
            if kw in sinonimos:
                keywords_expandidas.extend(sinonimos[kw])
        
        # Calcular scores
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            texto = chunk.page_content.lower()
            score = 0
            
            # Score por keywords
            for keyword in keywords_expandidas:
                count = texto.count(keyword)
                score += count * len(keyword) * 2
                
                # Bonus para aparição no início
                if keyword in texto[:300]:
                    score += len(keyword) * 5
            
            # Bonus por diversidade de keywords
            unique_kw_found = sum(1 for kw in keywords if kw in texto)
            score += unique_kw_found * 20
            
            # Bonus se tem indicadores de capítulo
            if re.search(r'cap[íi]tulo|chapter|seção|secao', texto):
                score += 30
            
            # Bonus por tamanho (chunks maiores tendem a ter mais contexto)
            score += len(chunk.page_content) * 0.01
            
            chunk_scores.append((score, i, chunk))
        
        # Ordenar e retornar
        chunk_scores.sort(reverse=True, key=lambda x: x[0])
        
        resultado = [chunk for score, idx, chunk in chunk_scores[:k]]
        
        # Log para debug
        logger.info(f"Busca inteligente - Query: '{query}', Keywords: {keywords}, Chunks retornados: {len(resultado)}")
        
        return resultado


def integrar_smart_retriever():
    """
    Integra o SmartRetriever no sistema existente.
    Adicione esta função no app.py após carregar o documento.
    """
    if 'documento_completo' not in st.session_state:
        return None
    
    if 'smart_retriever' not in st.session_state:
        retriever = SmartRetriever()
        documento = st.session_state['documento_completo']
        
        # Inicializar
        info = retriever.initialize_with_document(documento)
        st.session_state['smart_retriever'] = retriever
        
        logger.info(f"SmartRetriever inicializado: {info}")
        
        return info
    
    return st.session_state['smart_retriever']
