# No início do arquivo, após os imports:
from diagnostico import adicionar_interface_diagnostico, DocumentDiagnostic
from melhorias_recuperacao import SmartRetriever, integrar_smart_retriever

# Na função carrega_modelo, após processar o documento, adicione:
        # ... código existente ...
        
        # ADICIONAR AQUI - após processar documento
        status_text.text("🔍 Analisando estrutura...")
        progress_bar.progress(70)
        
        # Inicializar SmartRetriever
        try:
            retriever = SmartRetriever()
            estrutura_info = retriever.initialize_with_document(documento)
            st.session_state['smart_retriever'] = retriever
            
            logger.info(f"Estrutura identificada: {estrutura_info}")
            
            if estrutura_info['capitulos'] > 0:
                st.sidebar.success(f"✅ Identificados {estrutura_info['capitulos']} capítulos!")
        except Exception as e:
            logger.warning(f"Não foi possível analisar estrutura: {e}")
        
        # ... resto do código ...

# Na função processar_pergunta_com_documento, SUBSTITUIR a parte de recuperação:

def processar_pergunta_com_documento(
    input_usuario: str, 
    chain, 
    memoria: ConversationBufferMemory
) -> Generator[str, None, None]:
    """
    Processa perguntas usando chunks relevantes do documento de forma otimizada.
    """
    try:
        memory_manager = st.session_state.get('doc_memory_manager')
        if not memory_manager:
            yield "❌ Erro: Sistema não conseguiu acessar o documento. Por favor, tente recarregar."
            return
        
        # Obter configurações
        k_chunks = st.session_state.get('k_chunks', config.DEFAULT_K_CHUNKS)
        usando_doc_grande = st.session_state.get('usando_documento_grande', False)
        
        # Usar SmartRetriever se disponível
        smart_retriever = st.session_state.get('smart_retriever')
        
        if not usando_doc_grande:
            # Documento pequeno - usar completo
            pergunta_completa = input_usuario
        else:
            # Documento grande - usar recuperação inteligente
            if smart_retriever and 'doc_chunks' in st.session_state:
                # USAR SMART RETRIEVER
                chunks = st.session_state['doc_chunks']
                chunks_relevantes, contexto_estrutural = smart_retriever.retrieve_with_structure(
                    input_usuario, chunks, k=k_chunks
                )
            else:
                # Fallback para recuperação padrão
                chunks_relevantes = memory_manager.retrieve_relevant_chunks(
                    input_usuario, k=k_chunks
                )
                contexto_estrutural = ""
            
            if not chunks_relevantes:
                yield "⚠️ Não consegui encontrar informações relevantes no documento para responder sua pergunta."
                return
            
            # Montar contexto
            contexto_chunks = []
            for i, chunk in enumerate(chunks_relevantes):
                chunk_info = f"[Trecho {i+1}"
                if 'chunk_id' in chunk.metadata:
                    chunk_info += f" - Chunk #{chunk.metadata['chunk_id']}"
                chunk_info += "]\n" + chunk.page_content
                contexto_chunks.append(chunk_info)
            
            contexto_relevante = "\n\n---\n\n".join(contexto_chunks)
            
            # Montar prompt final
            if contexto_estrutural:
                prompt_adicional = f"""{contexto_estrutural}

TRECHOS RELEVANTES DO DOCUMENTO:
{contexto_relevante}


IMPORTANTE: Use TODAS as informações acima (estrutura e trechos) para responder de forma completa e precisa."""
            else:
                prompt_adicional = f"""TRECHOS RELEVANTES DO DOCUMENTO:
{contexto_relevante}


Use as informações acima para responder."""
            
            pergunta_completa = f"{prompt_adicional}\n\nPERGUNTA: {input_usuario}"
        
        # Debug info
        if st.session_state.get('show_debug', False):
            with st.expander("🔍 Debug - Informações"):
                st.text(f"Usando SmartRetriever: {smart_retriever is not None}")
                st.text(f"Contexto: {len(pergunta_completa)} caracteres")
                st.text(f"Tokens: ~{estimate_tokens(pergunta_completa)}")
                if usando_doc_grande and chunks_relevantes:
                    st.text(f"Chunks: {len(chunks_relevantes)}")
                    for i, chunk in enumerate(chunks_relevantes[:3]):
                        st.text(f"\nChunk {i+1} (ID: {chunk.metadata.get('chunk_id', 'N/A')}):")
                        st.code(chunk.page_content[:200])
        
        # Gerar resposta
        resposta_completa = ""
        for chunk in chain.stream({
            "input": pergunta_completa,
            "chat_history": memoria.buffer_as_messages
        }):
            if hasattr(chunk, 'content'):
                resposta_completa += chunk.content
            else:
                resposta_completa += str(chunk)
            yield resposta_completa
        
        # Estatísticas...
        # (resto do código permanece igual)
        
    except Exception as e:
        logger.error(f"Erro ao processar pergunta: {e}", exc_info=True)
        yield f"❌ Erro ao processar sua pergunta: {str(e)}"

# Na função sidebar(), adicionar antes do final:
    # === DIAGNÓSTICO ===
    adicionar_interface_diagnostico()
