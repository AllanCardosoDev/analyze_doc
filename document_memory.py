"""
Módulo avançado para gerenciamento de memória e processamento de documentos.
Implementa chunking, indexação vetorial opcional e recuperação inteligente.
"""
import logging
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import tempfile
import hashlib
import re
from typing import List, Dict, Optional, Any

from config import AppConfig, STOPWORDS_PT
from utils import calculate_file_hash, estimate_tokens

logger = logging.getLogger(__name__)


class DocumentMemoryManager:
    """
    Classe avançada para gerenciar memória e processamento de documentos.
    Suporta embeddings vetoriais opcionais para melhor recuperação.
    """
    
    def __init__(self, use_embeddings: bool = False):
        """
        Inicializa o gerenciador de memória de documentos.
        
        Args:
            use_embeddings: Se deve usar embeddings vetoriais (requer recursos adicionais)
        """
        self.use_embeddings = use_embeddings
        self.config = AppConfig()
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = None
        self.embedding_model = None
        
        if self.use_embeddings:
            self._init_embeddings()
    
    def _init_embeddings(self):
        """Inicializa o modelo de embeddings."""
        try:
            # Usando modelo multilíngue otimizado
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info("Modelo de embeddings inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar embeddings: {e}")
            self.use_embeddings = False
            st.warning("⚠️ Não foi possível carregar embeddings. Usando busca por palavras-chave.")
    
    def process_document(
        self, 
        documento: str, 
        tipo_documento: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Processa um documento, dividindo-o em chunks e opcionalmente indexando-o.
        
        Args:
            documento: Conteúdo do documento
            tipo_documento: Tipo do documento (PDF, DOCX, etc.)
            chunk_size: Tamanho dos chunks (usa padrão se None)
            chunk_overlap: Sobreposição entre chunks (usa padrão se None)
            
        Returns:
            dict: Metadados do processamento
        """
        # Usar valores padrão ou fornecidos
        chunk_size = chunk_size or self.config.DEFAULT_CHUNK_SIZE
        chunk_overlap = chunk_overlap or self.config.DEFAULT_CHUNK_OVERLAP
        
        # Calcular hash do documento
        doc_hash = calculate_file_hash(documento)
        
        # Contar páginas
        num_paginas = self._count_pages(documento, tipo_documento)
        
        # Dividir o documento em chunks
        chunks = self._split_document(documento, chunk_size, chunk_overlap)
        
        # Criar documentos do LangChain
        documents = []
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": tipo_documento,
                "chunk_id": i,
                "doc_hash": doc_hash,
                "chunk_count": len(chunks),
                "num_paginas": num_paginas,
                "chunk_size": len(chunk)
            }
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)
        
        # Armazenar os chunks na sessão
        st.session_state["doc_chunks"] = documents
        st.session_state["doc_hash"] = doc_hash
        st.session_state["num_paginas"] = num_paginas
        st.session_state["chunk_size_used"] = chunk_size
        st.session_state["chunk_overlap_used"] = chunk_overlap
        
        # Criar índice vetorial se embeddings estiverem habilitados
        index_created = False
        if self.use_embeddings and self.embedding_model:
            try:
                self.vector_store = FAISS.from_documents(documents, self.embedding_model)
                st.session_state["vector_store"] = self.vector_store
                index_created = True
                logger.info(f"Índice vetorial criado com {len(documents)} chunks")
            except Exception as e:
                logger.error(f"Erro ao criar índice vetorial: {e}")
                self.use_embeddings = False
        
        # Calcular estatísticas
        total_chars = sum(len(chunk) for chunk in chunks)
        avg_chunk_size = total_chars // len(chunks) if chunks else 0
        estimated_tokens = estimate_tokens(documento)
        
        return {
            "total_chunks": len(chunks),
            "doc_hash": doc_hash,
            "index_created": index_created,
            "tamanho_documento": len(documento),
            "num_paginas": num_paginas,
            "avg_chunk_size": avg_chunk_size,
            "estimated_tokens": estimated_tokens,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap
        }
    
    def _count_pages(self, documento: str, tipo_documento: str) -> int:
        """
        Conta o número aproximado de páginas no documento.
        
        Args:
            documento: Conteúdo do documento
            tipo_documento: Tipo do documento
            
        Returns:
            int: Número estimado de páginas
        """
        # Para documentos PDF com informação explícita de páginas
        if tipo_documento == "Pdf":
            # Buscar por padrões que indicam o número de páginas
            page_patterns = [
                r"Total de páginas:\s*(\d+)",
                r"Páginas:\s*(\d+)",
                r"(\d+)\s*páginas",
                r"página\s*\d+\s*de\s*(\d+)",
                r"--- Página\s+(\d+)\s+---"
            ]
            
            max_page = 0
            for pattern in page_patterns:
                matches = re.findall(pattern, documento, re.IGNORECASE)
                if matches:
                    try:
                        # Pegar o maior número encontrado
                        page_nums = [int(m) for m in matches]
                        max_page = max(max_page, max(page_nums))
                    except ValueError:
                        continue
            
            if max_page > 0:
                return max_page
        
        # Estimativa baseada no número de caracteres
        # ~3000 caracteres por página é uma estimativa razoável
        chars_per_page = 3000
        return max(1, len(documento) // chars_per_page)
    
    def _split_document(
        self, 
        documento: str, 
        chunk_size: int,
        chunk_overlap: int
    ) -> List[str]:
        """
        Divide o documento em chunks para processamento.
        
        Args:
            documento: Conteúdo do documento
            chunk_size: Tamanho dos chunks
            chunk_overlap: Sobreposição entre chunks
            
        Returns:
            list: Lista de chunks do documento
        """
        # Criar splitter com configuração otimizada
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        # Dividir documento
        chunks = text_splitter.split_text(documento)
        
        logger.info(f"Documento dividido em {len(chunks)} chunks")
        return chunks
    
    def retrieve_relevant_chunks(
        self, 
        query: str, 
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Recupera os chunks mais relevantes para uma consulta.
        Usa embeddings se disponível, caso contrário usa busca por palavras-chave.
        
        Args:
            query: Consulta do usuário
            k: Número de chunks a recuperar
            
        Returns:
            list: Lista de chunks relevantes
        """
        k = k or self.config.DEFAULT_K_CHUNKS
        
        if "doc_chunks" not in st.session_state:
            return []
        
        chunks = st.session_state["doc_chunks"]
        
        # Verificar se é uma pergunta sobre número de páginas
        if re.search(r'quantas\s+p[áa]ginas|n[úu]mero\s+de\s+p[áa]ginas', query.lower()):
            if "num_paginas" in st.session_state:
                num_paginas = st.session_state["num_paginas"]
                metadata = {"source": "info", "num_paginas": num_paginas}
                page_info = f"O documento possui {num_paginas} páginas."
                return [Document(page_content=page_info, metadata=metadata)]
        
        # Usar busca vetorial se disponível
        if self.use_embeddings and "vector_store" in st.session_state:
            try:
                vector_store = st.session_state["vector_store"]
                results = vector_store.similarity_search(query, k=k)
                logger.info(f"Recuperados {len(results)} chunks usando busca vetorial")
                return results
            except Exception as e:
                logger.error(f"Erro na busca vetorial: {e}")
                # Fallback para busca por palavras-chave
        
        # Busca por palavras-chave (fallback)
        return self._keyword_search(query, chunks, k)
    
    def _keyword_search(
        self, 
        query: str, 
        chunks: List[Document], 
        k: int
    ) -> List[Document]:
        """
        Implementa busca por palavras-chave melhorada.
        
        Args:
            query: Consulta do usuário
            chunks: Lista de chunks
            k: Número de chunks a retornar
            
        Returns:
            list: Chunks mais relevantes
        """
        # Normalizar a consulta
        query_norm = re.sub(r'[^\w\s]', '', query.lower())
        keywords = [
            word for word in query_norm.split() 
            if word not in STOPWORDS_PT and len(word) > 2
        ]
        
        if not keywords:
            # Se não há keywords válidas, retornar primeiros chunks
            return chunks[:k]
        
        # Calcular pontuação para cada chunk
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            texto = chunk.page_content.lower()
            score = 0
            
            # Pontuação baseada na frequência das palavras-chave
            for keyword in keywords:
                # Contar ocorrências exatas
                count = texto.count(keyword)
                # Palavras maiores têm mais peso
                score += count * len(keyword)
                
                # Bonus se a palavra aparece no início do chunk (mais relevante)
                if keyword in texto[:200]:
                    score += len(keyword) * 2
            
            # Bonus por diversidade de keywords
            unique_keywords_found = sum(1 for kw in keywords if kw in texto)
            score += unique_keywords_found * 10
            
            chunk_scores.append((i, score, chunk))
        
        # Ordenar por pontuação e obter os k chunks mais relevantes
        top_chunks = sorted(chunk_scores, key=lambda x: x[1], reverse=True)[:k]
        
        # Se nenhum chunk tiver pontuação, retornar os primeiros chunks
        if all(score == 0 for _, score, _ in top_chunks):
            logger.info("Nenhum chunk relevante encontrado, retornando primeiros chunks")
            return chunks[:k]
        
        # Retornar os chunks mais relevantes
        result_chunks = [chunk for _, _, chunk in top_chunks]
        logger.info(f"Recuperados {len(result_chunks)} chunks usando busca por palavras-chave")
        
        return result_chunks
    
    def get_document_preview(self, max_chars: int = 1500) -> str:
        """
        Gera um preview do documento para o contexto do modelo.
        
        Args:
            max_chars: Tamanho máximo do preview
            
        Returns:
            str: Preview do documento
        """
        if "documento_completo" not in st.session_state:
            return "Documento não disponível"
        
        documento = st.session_state["documento_completo"]
        
        # Se o documento for menor que o limite, retorna completo
        if len(documento) <= max_chars:
            return documento
        
        # Criar um preview inteligente
        # Pega início (50%), meio (25%) e fim (25%)
        inicio_size = max_chars // 2
        meio_size = max_chars // 4
        fim_size = max_chars // 4
        
        inicio = documento[:inicio_size]
        
        meio_pos = len(documento) // 2
        meio_start = max(0, meio_pos - meio_size // 2)
        meio_end = min(len(documento), meio_pos + meio_size // 2)
        meio = documento[meio_start:meio_end]
        
        fim = documento[-fim_size:]
        
        preview = f"{inicio}\n\n[...]\n\n{meio}\n\n[...]\n\n{fim}"
        
        return preview
    
    def get_document_info(self) -> Dict[str, Any]:
        """
        Retorna informações detalhadas sobre o documento.
        
        Returns:
            dict: Informações do documento
        """
        info = {
            "tamanho": st.session_state.get("tamanho_documento", 0),
            "tipo": st.session_state.get("tipo_arquivo", "Desconhecido"),
            "num_paginas": st.session_state.get("num_paginas", 0),
            "num_chunks": len(st.session_state.get("doc_chunks", [])),
            "chunk_size": st.session_state.get("chunk_size_used", self.config.DEFAULT_CHUNK_SIZE),
            "chunk_overlap": st.session_state.get("chunk_overlap_used", self.config.DEFAULT_CHUNK_OVERLAP),
            "doc_hash": st.session_state.get("doc_hash", ""),
            "using_embeddings": self.use_embeddings and "vector_store" in st.session_state,
            "estimated_tokens": estimate_tokens(str(st.session_state.get("tamanho_documento", 0)))
        }
        return info
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas sobre o sistema de recuperação.
        
        Returns:
            dict: Estatísticas de recuperação
        """
        stats = {
            "total_queries": st.session_state.get("total_queries", 0),
            "using_vector_search": self.use_embeddings and "vector_store" in st.session_state,
            "avg_chunks_retrieved": st.session_state.get("avg_chunks_retrieved", 0)
        }
        return stats
    
    def cleanup(self):
        """Limpa os arquivos temporários criados."""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Arquivos temporários limpos")
        except Exception as e:
            logger.error(f"Erro ao limpar arquivos temporários: {e}")
