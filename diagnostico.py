"""
Ferramenta de diagnóstico e correção para problemas de recuperação de documentos.
"""
import streamlit as st
import re
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DocumentDiagnostic:
    """Diagnóstica problemas na recuperação de informações do documento."""
    
    @staticmethod
    def analizar_estrutura_documento(documento: str) -> Dict[str, Any]:
        """
        Analisa a estrutura do documento para identificar capítulos, seções, etc.
        
        Args:
            documento: Conteúdo completo do documento
            
        Returns:
            dict: Estrutura identificada
        """
        estrutura = {
            'capitulos': [],
            'secoes': [],
            'paginas': [],
            'indices': []
        }
        
        # Padrões para identificar capítulos
        padroes_capitulos = [
            r'Cap[íi]tulo\s+(\d+)[\s:.-]+(.+?)(?=\n|$)',
            r'CAPÍTULO\s+(\d+)[\s:.-]+(.+?)(?=\n|$)',
            r'Chapter\s+(\d+)[\s:.-]+(.+?)(?=\n|$)',
            r'^\s*(\d+)\s*[-–—.]\s*(.+?)(?=\n|$)',
            r'^\s*(\d+)\.\s+(.+?)(?=\n|$)',
            r'---\s*Página\s+(\d+)\s*---'
        ]
        
        linhas = documento.split('\n')
        for i, linha in enumerate(linhas):
            linha_limpa = linha.strip()
            
            # Identificar capítulos
            for padrao in padroes_capitulos:
                match = re.search(padrao, linha_limpa, re.IGNORECASE)
                if match:
                    if 'Página' in padrao:
                        estrutura['paginas'].append({
                            'numero': match.group(1),
                            'linha': i,
                            'contexto': '\n'.join(linhas[max(0,i-2):min(len(linhas),i+10)])
                        })
                    else:
                        estrutura['capitulos'].append({
                            'numero': match.group(1),
                            'titulo': match.group(2).strip(),
                            'linha': i,
                            'contexto': '\n'.join(linhas[i:min(len(linhas),i+20)])
                        })
                    break
            
            # Identificar índice/sumário
            if re.search(r'(sumário|índice|contents|table of contents)', linha_limpa, re.IGNORECASE):
                estrutura['indices'].append({
                    'linha': i,
                    'contexto': '\n'.join(linhas[i:min(len(linhas),i+50)])
                })
        
        return estrutura
    
    @staticmethod
    def criar_mapa_documento(documento: str, estrutura: Dict[str, Any]) -> str:
        """
        Cria um mapa estruturado do documento para incluir no contexto.
        
        Args:
            documento: Conteúdo do documento
            estrutura: Estrutura identificada
            
        Returns:
            str: Mapa do documento
        """
        mapa = "=== MAPA DO DOCUMENTO ===\n\n"
        
        # Informações gerais
        mapa += f"Total de caracteres: {len(documento):,}\n"
        mapa += f"Páginas identificadas: {len(estrutura['paginas'])}\n"
        mapa += f"Capítulos identificados: {len(estrutura['capitulos'])}\n\n"
        
        # Listar capítulos encontrados
        if estrutura['capitulos']:
            mapa += "CAPÍTULOS ENCONTRADOS:\n"
            for cap in estrutura['capitulos'][:20]:  # Limitar a 20
                mapa += f"• Capítulo {cap['numero']}: {cap['titulo'][:100]}\n"
            mapa += "\n"
        
        # Listar primeiras páginas
        if estrutura['paginas']:
            mapa += "PÁGINAS IDENTIFICADAS:\n"
            mapa += f"De página {estrutura['paginas'][0]['numero']} até {estrutura['paginas'][-1]['numero']}\n\n"
        
        # Incluir índice se encontrado
        if estrutura['indices']:
            mapa += "ÍNDICE/SUMÁRIO ENCONTRADO:\n"
            mapa += estrutura['indices'][0]['contexto'][:1000] + "\n\n"
        
        return mapa
    
    @staticmethod
    def extrair_capitulo_especifico(
        documento: str, 
        numero_capitulo: int,
        estrutura: Dict[str, Any]
    ) -> str:
        """
        Extrai o conteúdo completo de um capítulo específico.
        
        Args:
            documento: Conteúdo do documento
            numero_capitulo: Número do capítulo
            estrutura: Estrutura do documento
            
        Returns:
            str: Conteúdo do capítulo
        """
        capitulos = estrutura['capitulos']
        
        # Encontrar o capítulo solicitado
        capitulo_atual = None
        proximo_capitulo = None
        
        for i, cap in enumerate(capitulos):
            if int(cap['numero']) == numero_capitulo:
                capitulo_atual = cap
                if i + 1 < len(capitulos):
                    proximo_capitulo = capitulos[i + 1]
                break
        
        if not capitulo_atual:
            return f"Capítulo {numero_capitulo} não encontrado."
        
        # Extrair conteúdo entre este capítulo e o próximo
        linhas = documento.split('\n')
        inicio = capitulo_atual['linha']
        
        if proximo_capitulo:
            fim = proximo_capitulo['linha']
        else:
            # Se é o último capítulo, ir até o fim
            fim = len(linhas)
        
        conteudo_capitulo = '\n'.join(linhas[inicio:fim])
        
        return conteudo_capitulo
    
    @staticmethod
    def testar_recuperacao_chunks(query: str, chunks: List, top_k: int = 5) -> List[Dict]:
        """
        Testa e mostra quais chunks seriam recuperados para uma query.
        
        Args:
            query: Pergunta do usuário
            chunks: Lista de chunks
            top_k: Quantos chunks mostrar
            
        Returns:
            list: Chunks com pontuações
        """
        from config import STOPWORDS_PT
        
        # Normalizar query
        query_norm = re.sub(r'[^\w\s]', '', query.lower())
        keywords = [
            word for word in query_norm.split() 
            if word not in STOPWORDS_PT and len(word) > 2
        ]
        
        # Calcular pontuações
        resultados = []
        for i, chunk in enumerate(chunks):
            texto = chunk.page_content.lower()
            score = 0
            keywords_found = []
            
            for keyword in keywords:
                count = texto.count(keyword)
                if count > 0:
                    score += count * len(keyword)
                    keywords_found.append(keyword)
                
                # Bonus para início do chunk
                if keyword in texto[:200]:
                    score += len(keyword) * 2
            
            resultados.append({
                'chunk_id': i,
                'score': score,
                'keywords_found': keywords_found,
                'preview': chunk.page_content[:200] + "...",
                'metadata': chunk.metadata,
                'full_content': chunk.page_content
            })
        
        # Ordenar por score
        resultados_ordenados = sorted(resultados, key=lambda x: x['score'], reverse=True)
        
        return resultados_ordenados[:top_k]


def adicionar_interface_diagnostico():
    """Adiciona interface de diagnóstico na sidebar."""
    
    if 'documento_completo' not in st.session_state:
        return
    
    with st.sidebar.expander("🔍 Diagnóstico Avançado"):
        st.markdown("### Analisar Estrutura do Documento")
        
        if st.button("Analisar Estrutura", use_container_width=True):
            with st.spinner("Analisando documento..."):
                documento = st.session_state['documento_completo']
                
                diagnostico = DocumentDiagnostic()
                estrutura = diagnostico.analizar_estrutura_documento(documento)
                
                # Salvar na sessão
                st.session_state['estrutura_documento'] = estrutura
                
                # Mostrar resultados
                st.success("✅ Análise concluída!")
                st.json({
                    'Capítulos encontrados': len(estrutura['capitulos']),
                    'Páginas identificadas': len(estrutura['paginas']),
                    'Índices encontrados': len(estrutura['indices'])
                })
        
        # Testar recuperação
        st.markdown("### Testar Recuperação")
        query_teste = st.text_input("Query de teste:", "primeiro capítulo")
        
        if st.button("Testar", use_container_width=True):
            if 'doc_chunks' in st.session_state:
                chunks = st.session_state['doc_chunks']
                diagnostico = DocumentDiagnostic()
                
                resultados = diagnostico.testar_recuperacao_chunks(
                    query_teste, chunks, top_k=3
                )
                
                st.markdown("**Chunks Recuperados:**")
                for i, res in enumerate(resultados):
                    with st.expander(f"Chunk {i+1} - Score: {res['score']}"):
                        st.write(f"**Keywords encontradas:** {', '.join(res['keywords_found'])}")
                        st.write(f"**Chunk ID:** {res['chunk_id']}")
                        st.write(f"**Metadata:** {res['metadata']}")
                        st.text_area("Preview:", res['preview'], height=100)
        
        # Extrair capítulo específico
        if 'estrutura_documento' in st.session_state:
            estrutura = st.session_state['estrutura_documento']
            if estrutura['capitulos']:
                st.markdown("### Extrair Capítulo")
                cap_num = st.number_input(
                    "Número do capítulo:",
                    min_value=1,
                    max_value=50,
                    value=1
                )
                
                if st.button("Extrair", use_container_width=True):
                    documento = st.session_state['documento_completo']
                    diagnostico = DocumentDiagnostic()
                    
                    conteudo = diagnostico.extrair_capitulo_especifico(
                        documento, cap_num, estrutura
                    )
                    
                    st.text_area("Conteúdo do capítulo:", conteudo[:2000], height=300)
