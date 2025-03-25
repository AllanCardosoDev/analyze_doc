"""
Módulo de colaboração para o Analyse Doc.
Contém funções para gerenciar compartilhamento, comentários, histórico de
versões e colaboração entre usuários.
"""

import uuid
import json
import datetime
from typing import Dict, List, Any, Optional
import difflib
import streamlit as st

def cria_comentario(usuario_id: str, documento_id: str, texto: str, tipo: str = "comentário",
                   posicao: Optional[int] = None) -> Dict[str, Any]:
    """
    Cria um novo comentário ou anotação.
    
    Args:
        usuario_id: ID do usuário que criou o comentário
        documento_id: ID do documento relacionado
        texto: Conteúdo do comentário
        tipo: Tipo de comentário (comentário, anotação, tarefa)
        posicao: Posição no documento (opcional)
        
    Returns:
        Dados do comentário criado
    """
    comentario_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    comentario = {
        "id": comentario_id,
        "usuario_id": usuario_id,
        "documento_id": documento_id,
        "texto": texto,
        "tipo": tipo,
        "timestamp": timestamp,
        "posicao": posicao,
        "status": "ativo",
        "respostas": []
    }
    
    return comentario

def responde_comentario(comentario_id: str, usuario_id: str, texto: str,
                       comentarios: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Adiciona uma resposta a um comentário existente.
    
    Args:
        comentario_id: ID do comentário original
        usuario_id: ID do usuário que está respondendo
        texto: Conteúdo da resposta
        comentarios: Lista de comentários existentes
        
    Returns:
        Lista atualizada de comentários ou erro
    """
    # Encontra o comentário original
    for i, comentario in enumerate(comentarios):
        if comentario["id"] == comentario_id:
            # Cria a resposta
            resposta = {
                "id": str(uuid.uuid4()),
                "usuario_id": usuario_id,
                "texto": texto,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            # Adiciona a resposta ao comentário
            comentarios[i]["respostas"].append(resposta)
            return {"status": "sucesso", "comentarios": comentarios}
    
    # Se não encontrou o comentário
    return {"status": "erro", "mensagem": "Comentário não encontrado"}

def compartilha_documento(documento_id: str, usuario_origem: str, usuario_destino: str,
                         nivel_acesso: str = "leitura") -> Dict[str, Any]:
    """
    Compartilha um documento com outro usuário.
    
    Args:
        documento_id: ID do documento
        usuario_origem: ID do usuário que está compartilhando
        usuario_destino: ID ou email do usuário que receberá acesso
        nivel_acesso: Nível de acesso (leitura, comentário, edição)
        
    Returns:
        Dados do compartilhamento
    """
    compartilhamento_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    compartilhamento = {
        "id": compartilhamento_id,
        "documento_id": documento_id,
        "usuario_origem": usuario_origem,
        "usuario_destino": usuario_destino,
        "nivel_acesso": nivel_acesso,
        "timestamp": timestamp,
        "status": "ativo"
    }
    
    # Código para enviar notificação (e-mail/notificação)
    # seria implementado aqui em uma versão completa
    
    return compartilhamento

def cria_versao(documento_id: str, usuario_id: str, conteudo: str, 
               descricao: Optional[str] = None) -> Dict[str, Any]:
    """
    Cria uma nova versão do documento.
    
    Args:
        documento_id: ID do documento
        usuario_id: ID do usuário que está criando a versão
        conteudo: Conteúdo do documento
        descricao: Descrição da versão (opcional)
        
    Returns:
        Dados da versão criada
    """
    versao_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    versao = {
        "id": versao_id,
        "documento_id": documento_id,
        "usuario_id": usuario_id,
        "conteudo": conteudo,
        "descricao": descricao or f"Versão criada em {timestamp}",
        "timestamp": timestamp,
        "hash": hash(conteudo)  # Simples, uma implementação real usaria algo melhor
    }
    
    return versao

def compara_versoes(versao1: Dict[str, Any], versao2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compara duas versões de um documento e retorna as diferenças.
    
    Args:
        versao1: Dados da primeira versão
        versao2: Dados da segunda versão
        
    Returns:
        Diferenças entre as versões
    """
    # Obtém o conteúdo das versões
    conteudo1 = versao1.get("conteudo", "")
    conteudo2 = versao2.get("conteudo", "")
    
    # Divide o texto em linhas
    linhas1 = conteudo1.splitlines()
    linhas2 = conteudo2.splitlines()
    
    # Obtém as diferenças
    diff = list(difflib.unified_diff(
        linhas1, linhas2,
        fromfile=f"Versão {versao1.get('id', 'anterior')}",
        tofile=f"Versão {versao2.get('id', 'atual')}",
        lineterm="",
        n=3
    ))
    
    # Estatísticas simples
    linhas_adicionadas = sum(1 for linha in diff if linha.startswith('+') and not linha.startswith('+++'))
    linhas_removidas = sum(1 for linha in diff if linha.startswith('-') and not linha.startswith('---'))
    
    return {
        "versao1": versao1.get("id"),
        "versao2": versao2.get("id"),
        "timestamp1": versao1.get("timestamp"),
        "timestamp2": versao2.get("timestamp"),
        "diff": diff,
        "estatisticas": {
            "linhas_adicionadas": linhas_adicionadas,
            "linhas_removidas": linhas_removidas,
            "total_alteracoes": linhas_adicionadas + linhas_removidas
        }
    }

def cria_tarefa(documento_id: str, usuario_criador: str, usuario_responsavel: str,
              descricao: str, prazo: Optional[str] = None) -> Dict[str, Any]:
    """
    Cria uma nova tarefa relacionada a um documento.
    
    Args:
        documento_id: ID do documento
        usuario_criador: ID do usuário que criou a tarefa
        usuario_responsavel: ID do usuário responsável pela tarefa
        descricao: Descrição da tarefa
        prazo: Data limite para conclusão (formato ISO)
        
    Returns:
        Dados da tarefa criada
    """
    tarefa_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    # Converte o prazo para datetime se fornecido
    prazo_dt = None
    if prazo:
        try:
            prazo_dt = datetime.datetime.fromisoformat(prazo)
        except ValueError:
            prazo_dt = None
    
    tarefa = {
        "id": tarefa_id,
        "documento_id": documento_id,
        "usuario_criador": usuario_criador,
        "usuario_responsavel": usuario_responsavel,
        "descricao": descricao,
        "prazo": prazo,
        "status": "pendente",
        "criado_em": timestamp,
        "atualizado_em": timestamp,
        "comentarios": []
    }
    
    return tarefa

def atualiza_status_tarefa(tarefa_id: str, novo_status: str, 
                          tarefas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Atualiza o status de uma tarefa.
    
    Args:
        tarefa_id: ID da tarefa
        novo_status: Novo status (pendente, em progresso, concluída, cancelada)
        tarefas: Lista de tarefas existentes
        
    Returns:
        Lista atualizada de tarefas ou erro
    """
    # Verifica se o status é válido
    status_validos = ["pendente", "em progresso", "concluída", "cancelada"]
    if novo_status not in status_validos:
        return {"status": "erro", "mensagem": f"Status inválido. Use um dos: {', '.join(status_validos)}"}
    
    # Encontra a tarefa
    for i, tarefa in enumerate(tarefas):
        if tarefa["id"] == tarefa_id:
            # Atualiza o status
            tarefas[i]["status"] = novo_status
            tarefas[i]["atualizado_em"] = datetime.datetime.now().isoformat()
            return {"status": "sucesso", "tarefas": tarefas}
    
    # Se não encontrou a tarefa
    return {"status": "erro", "mensagem": "Tarefa não encontrada"}

def gera_link_compartilhamento(documento_id: str, usuario_id: str, 
                            nivel_acesso: str = "leitura",
                            expiracao_dias: int = 7) -> Dict[str, Any]:
    """
    Gera um link de compartilhamento para o documento.
    
    Args:
        documento_id: ID do documento
        usuario_id: ID do usuário que está compartilhando
        nivel_acesso: Nível de acesso (leitura, comentário)
        expiracao_dias: Dias até a expiração do link
        
    Returns:
        Dados do link de compartilhamento
    """
    link_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now()
    expiracao = timestamp + datetime.timedelta(days=expiracao_dias)
    
    link_dados = {
        "id": link_id,
        "documento_id": documento_id,
        "usuario_id": usuario_id,
        "nivel_acesso": nivel_acesso,
        "criado_em": timestamp.isoformat(),
        "expira_em": expiracao.isoformat(),
        "acessos": 0,
        "status": "ativo"
    }
    
    # Cria o link
    # Em uma implementação real, isso seria uma URL com o token
    link_url = f"https://analysedoc.example.com/s/{link_id}"
    
    return {
        "url": link_url,
        "dados": link_dados
    }

def renderiza_comentarios(comentarios: List[Dict[str, Any]], usuario_atual: str) -> None:
    """
    Renderiza comentários na interface Streamlit.
    
    Args:
        comentarios: Lista de comentários
        usuario_atual: ID do usuário atual
    """
    if not comentarios:
        st.info("Sem comentários ou anotações")
        return
    
    for comentario in comentarios:
        with st.expander(f"{comentario['tipo'].capitalize()} - {comentario['timestamp'][:16].replace('T', ' ')}"):
            # Informações do cabeçalho
            col1, col2 = st.columns([3, 1])
            col1.markdown(f"**Usuário:** {comentario['usuario_id']}")
            
            # Ícones para o tipo de comentário
            if comentario['tipo'] == "comentário":
                icone = "💬"
            elif comentario['tipo'] == "anotação":
                icone = "📝"
            elif comentario['tipo'] == "tarefa":
                icone = "✅"
            else:
                icone = "📌"
            
            col2.markdown(f"{icone} {comentario['tipo'].capitalize()}")
            
            # Conteúdo principal
            st.markdown(comentario['texto'])
            
            # Respostas, se houver
            if comentario.get('respostas'):
                st.divider()
                st.markdown("**Respostas:**")
                
                for resposta in comentario['respostas']:
                    st.markdown(f"**{resposta['usuario_id']}** ({resposta['timestamp'][:16].replace('T', ' ')})")
                    st.markdown(resposta['texto'])
                    st.markdown("---")
            
            # Campo para adicionar resposta
            with st.form(f"resposta_{comentario['id']}"):
                resposta_texto = st.text_area("Responder", key=f"resp_{comentario['id']}")
                enviar = st.form_submit_button("Enviar resposta")
                
                if enviar and resposta_texto:
                    st.session_state["comments_updated"] = True
                    # O código para salvar a resposta estaria aqui
                    # em uma implementação completa

def renderiza_tarefas(tarefas: List[Dict[str, Any]], usuario_atual: str) -> None:
    """
    Renderiza tarefas na interface Streamlit.
    
    Args:
        tarefas: Lista de tarefas
        usuario_atual: ID do usuário atual
    """
    if not tarefas:
        st.info("Sem tarefas associadas")
        return
    
    # Agrupa por status
    tarefas_por_status = {
        "pendente": [],
        "em progresso": [],
        "concluída": [],
        "cancelada": []
    }
    
    for tarefa in tarefas:
        status = tarefa.get("status", "pendente")
        tarefas_por_status[status].append(tarefa)
    
    # Renderiza por status
    tabs = st.tabs(["Pendentes", "Em Progresso", "Concluídas", "Canceladas"])
    
    with tabs[0]:
        for tarefa in tarefas_por_status["pendente"]:
            st.warning(tarefa["descricao"])
            col1, col2 = st.columns([3, 1])
            col1.caption(f"Responsável: {tarefa['usuario_responsavel']}")
            if tarefa.get("prazo"):
                col2.caption(f"Prazo: {tarefa['prazo'][:10]}")
            
            # Botões de ação
            if tarefa["usuario_responsavel"] == usuario_atual:
                bt1, bt2 = st.columns(2)
                if bt1.button("Iniciar", key=f"iniciar_{tarefa['id']}"):
                    st.session_state["task_updated"] = True
                    # Código para atualizar tarefa iria aqui
                
                if bt2.button("Cancelar", key=f"cancelar_{tarefa['id']}"):
                    st.session_state["task_updated"] = True
                    # Código para cancelar tarefa iria aqui
    
    with tabs[1]:
        for tarefa in tarefas_por_status["em progresso"]:
            st.info(tarefa["descricao"])
            col1, col2 = st.columns([3, 1])
            col1.caption(f"Responsável: {tarefa['usuario_responsavel']}")
            if tarefa.get("prazo"):
                col2.caption(f"Prazo: {tarefa['prazo'][:10]}")
            
            # Botão de conclusão
            if tarefa["usuario_responsavel"] == usuario_atual:
                if st.button("Concluir", key=f"concluir_{tarefa['id']}"):
                    st.session_state["task_updated"] = True
                    # Código para concluir tarefa iria aqui
    
    with tabs[2]:
        for tarefa in tarefas_por_status["concluída"]:
            st.success(tarefa["descricao"])
            st.caption(f"Concluída por: {tarefa['usuario_responsavel']}")
    
    with tabs[3]:
        for tarefa in tarefas_por_status["cancelada"]:
            st.error(tarefa["descricao"])
            st.caption(f"Cancelada em: {tarefa['atualizado_em'][:10]}")

def notificar_usuario(usuario_id: str, tipo: str, mensagem: str, 
                    referencia_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Cria uma notificação para um usuário.
    
    Args:
        usuario_id: ID do usuário a notificar
        tipo: Tipo de notificação (compartilhamento, tarefa, comentário)
        mensagem: Conteúdo da notificação
        referencia_id: ID do item relacionado (opcional)
        
    Returns:
        Dados da notificação
    """
    notificacao_id = str(uuid.uuid4())
    timestamp = datetime.datetime.now().isoformat()
    
    notificacao = {
        "id": notificacao_id,
        "usuario_id": usuario_id,
        "tipo": tipo,
        "mensagem": mensagem,
        "referencia_id": referencia_id,
        "timestamp": timestamp,
        "lida": False
    }
    
    # Em uma implementação real, enviaria a notificação por email/push
    
    return notificacao
