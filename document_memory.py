"""
Módulo para gerenciamento de memória e processamento de documentos grandes.
Implementa técnicas de chunking, indexação e recuperação para permitir análise
de documentos de qualquer tamanho.
"""

import logging
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import tempfile
import hashlib
import re

# Configuração de logging
logger = logging.getLogger(__name__)

class DocumentMemoryManager:
    """Classe para gerenciar a memória e processamento de documentos grandes."""
    
    def __init__(self, embedding_model=None):
        """
        Inicializa o gerenciador de memória de documentos.
        
        Args:
            embedding_model: Modelo de embeddings para indexação (opcional)
        """
        self.embedding_model = embedding_model
        self.temp_dir = tempfile.mkdtemp()
        
    def process_document(self, documento, tipo_documento):
        """
        Processa um documento, dividindo-o em chunks e indexando-o.
        
        Args:
            documento (str): Conteúdo do documento
            tipo_documento (str): Tipo do documento (PDF, DOCX, etc.)
            
        Returns:
            dict: Metadados do processamento
        """
        # Calcular hash do documento para identificação única
        doc_hash = hashlib.md5(documento.encode()).hexdigest()
        
        # Dividir o documento em chunks
        chunks = self._split_document(documento)
        
        # Criar documentos do LangChain
        documents = []
        for i, chunk in enumerate(chunks):
            # Criar metadados para o chunk
            metadata = {
                "source": tipo_documento,
                "chunk_id": i,
                "doc_hash": doc_hash,
                "chunk_count": len(chunks)
            }
            
            # Criar documento
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)
        
        # Armazenar os chunks na sessão
        st.session_state["doc_chunks"] = documents
        st.session_state["doc_hash"] = doc_hash
        
        # Retornar metadados do processamento
        return {
            "total_chunks": len(chunks),
            "doc_hash": doc_hash,
            "index_created": False,
            "tamanho_documento": len(documento)
        }
    
    def _split_document(self, documento):
        """
        Divide o documento em chunks para processamento.
        
        Args:
            documento (str): Conteúdo do documento
            
        Returns:
            list: Lista de chunks do documento
        """
        # Criar splitter com configuração otimizada
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,           # 4000 caracteres por chunk
            chunk_overlap=400,         # 10% de sobreposição
            separators=["\n\n", "\n", ". ", " ", ""],  # Priorizar quebras de parágrafo
            length_function=len
        )
        
        # Dividir documento
        chunks = text_splitter.split_text(documento)
        return chunks
    
    def retrieve_relevant_chunks(self, query, k=4):
        """
        Recupera os chunks mais relevantes para uma consulta.
        Se não temos vetores, usamos correspondência de palavras-chave simples.
        
        Args:
            query (str): Consulta do usuário
            k (int): Número de chunks a recuperar
            
        Returns:
            list: Lista de chunks relevantes
        """
        if "doc_chunks" not in st.session_state:
            return []
            
        # Implementação básica sem vetores: uso de correspondência de palavras-chave
        chunks = st.session_state["doc_chunks"]
        
        # Extrair palavras-chave da consulta (remover stopwords comuns)
        stopwords = {'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da', 'dos', 'das', 
                    'em', 'no', 'na', 'nos', 'nas', 'por', 'para', 'com', 'sem', 'sob', 'sobre', 'e', 
                    'ou', 'que', 'porque', 'quando', 'onde', 'como', 'qual', 'quais', 'é', 'são'}
        
        # Normalizar a consulta (remover pontuação e converter para minúsculas)
        query_norm = re.sub(r'[^\w\s]', '', query.lower())
        keywords = [word for word in query_norm.split() if word not in stopwords and len(word) > 2]
        
        # Calcular pontuação para cada chunk
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            texto = chunk.page_content.lower()
            score = 0
            # Pontuação baseada na frequência das palavras-chave
            for keyword in keywords:
                score += texto.count(keyword) * len(keyword)  # Palavras maiores têm mais peso
            chunk_scores.append((i, score))
        
        # Ordenar por pontuação e obter os k chunks mais relevantes
        top_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:k]
        
        # Se nenhum chunk tiver pontuação, retornar os primeiros chunks
        if all(score == 0 for _, score in top_chunks):
            return chunks[:k]
        
        # Retornar os chunks mais relevantes
        return [chunks[i] for i, _ in top_chunks]
    
    def get_document_preview(self, max_chars=3000):
        """
        Gera um preview do documento para o contexto do modelo.
        
        Args:
            max_chars (int): Tamanho máximo do preview
            
        Returns:
            str: Preview do documento
        """
        if "documento_completo" not in st.session_state:
            return "Documento não disponível"
        
        documento = st.session_state["documento_completo"]
        
        # Se o documento for menor que o limite, retorna completo
        if len(documento) <= max_chars:
            return documento
        
        # Caso contrário, cria um preview com início, meio e fim
        inicio = documento[:max_chars // 3]
        
        meio_pos = len(documento) // 2
        meio = documento[meio_pos - (max_chars // 6):meio_pos + (max_chars // 6)]
        
        fim = documento[-max_chars // 3:]
        
        return f"{inicio}\n\n[...]\n\n{meio}\n\n[...]\n\n{fim}"
    
    def cleanup(self):
        """Limpa os arquivos temporários criados."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logger.error(f"Erro ao limpar arquivos temporários: {e}")
