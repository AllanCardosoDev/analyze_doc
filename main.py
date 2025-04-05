"""
Arquivo principal para a aplicação Analyse Doc.
Ponto de entrada que inicializa o aplicativo Streamlit.
"""

import streamlit as st
from dotenv import load_dotenv
import os
import app
from document_memory import DocumentMemoryManager

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar aplicação com debug se configurado
debug = os.getenv("DEBUG", "False").lower() == "true"
if debug:
    import logging
    logging.basicConfig(level=logging.DEBUG)
    st.set_option('deprecation.showfileUploaderEncoding', False)

# Inicializar a sessão se for o primeiro carregamento
if "inicializado" not in st.session_state:
    st.session_state["inicializado"] = True
    st.session_state["mostrar_resumo"] = False
    
    # Configurações padrão para o resumo
    st.session_state["gerar_resumo"] = True
    st.session_state["max_resumo_length"] = 1500
    st.session_state["usar_llm_resumo"] = True
    st.session_state["incluir_topicos"] = True
    st.session_state["incluir_termos"] = True
    st.session_state["analisar_estrutura"] = False
    
    # Configurações padrão para processamento de documentos grandes
    st.session_state["max_tamanho_padrao"] = 50000
    st.session_state["sempre_usar_processamento_avancado"] = False
    
    # Inicializar o gerenciador de memória de documentos
    st.session_state["doc_memory_manager"] = DocumentMemoryManager()

def main():
    """Função principal que executa a aplicação."""
    # Executar a aplicação
    app.main()

if __name__ == "__main__":
    main()
