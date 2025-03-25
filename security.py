"""
Módulo de segurança para o Analyse Doc.
Contém funções para anonimização, criptografia, gerenciamento de dados e
controle de acesso.
"""

import re
import os
import json
import hashlib
import uuid
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Importação opcional para criptografia
try:
    from cryptography.fernet import Fernet
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# Importamos a função de anonimização do módulo loaders para manter compatibilidade
from loaders import anonimizar_texto

def gera_chave_api():
    """Gera uma chave de API aleatória."""
    return str(uuid.uuid4())

def hash_senha(senha: str) -> str:
    """
    Cria um hash seguro de senha.
    
    Args:
        senha: A senha em texto plano
        
    Returns:
        Hash da senha
    """
    salt = os.urandom(32)
    hash = hashlib.pbkdf2_hmac(
        'sha256',
        senha.encode('utf-8'),
        salt,
        100000
    )
    return salt.hex() + hash.hex()

def verifica_senha(senha: str, hash_armazenado: str) -> bool:
    """
    Verifica se uma senha corresponde ao hash armazenado.
    
    Args:
        senha: A senha em texto plano para verificar
        hash_armazenado: O hash armazenado
        
    Returns:
        True se a senha estiver correta, False caso contrário
    """
    # Extrai o salt
    salt = bytes.fromhex(hash_armazenado[:64])
    # Extrai o hash armazenado
    hash_stored = hash_armazenado[64:]
    
    # Calcula o hash da senha fornecida
    hash_calculado = hashlib.pbkdf2_hmac(
        'sha256',
        senha.encode('utf-8'),
        salt,
        100000
    ).hex()
    
    # Compara os hashes
    return hash_calculado == hash_stored

def criptografa_documento(texto: str, chave: Optional[str] = None) -> Dict[str, str]:
    """
    Criptografa o conteúdo de um documento.
    
    Args:
        texto: O texto a ser criptografado
        chave: Chave de criptografia (opcional)
        
    Returns:
        Dicionário com texto criptografado e chave
    """
    if not CRYPTO_AVAILABLE:
        return {
            "status": "erro",
            "mensagem": "Biblioteca de criptografia não disponível",
            "texto": texto
        }
    
    try:
        # Gera uma chave se não for fornecida
        if not chave:
            key = Fernet.generate_key()
        else:
            # Se fornecida, verifica se é uma chave Fernet válida
            try:
                key = chave.encode() if isinstance(chave, str) else chave
                Fernet(key)  # Testa se é uma chave válida
            except:
                # Se não for válida, gera uma nova
                key = Fernet.generate_key()
        
        # Criptografa o texto
        f = Fernet(key)
        texto_bytes = texto.encode('utf-8')
        texto_criptografado = f.encrypt(texto_bytes)
        
        return {
            "status": "sucesso",
            "texto_criptografado": texto_criptografado.decode('utf-8'),
            "chave": key.decode('utf-8') if isinstance(key, bytes) else key
        }
    except Exception as e:
        return {
            "status": "erro",
            "mensagem": f"Erro ao criptografar: {str(e)}",
            "texto": texto
        }

def descriptografa_documento(texto_criptografado: str, chave: str) -> Dict[str, str]:
    """
    Descriptografa o conteúdo de um documento.
    
    Args:
        texto_criptografado: O texto criptografado
        chave: Chave de criptografia
        
    Returns:
        Dicionário com texto descriptografado
    """
    if not CRYPTO_AVAILABLE:
        return {
            "status": "erro",
            "mensagem": "Biblioteca de criptografia não disponível",
            "texto": texto_criptografado
        }
    
    try:
        # Prepara a chave
        key = chave.encode() if isinstance(chave, str) else chave
        
        # Descriptografa o texto
        f = Fernet(key)
        texto_bytes = texto_criptografado.encode('utf-8') if isinstance(texto_criptografado, str) else texto_criptografado
        texto_descriptografado = f.decrypt(texto_bytes)
        
        return {
            "status": "sucesso",
            "texto": texto_descriptografado.decode('utf-8')
        }
    except Exception as e:
        return {
            "status": "erro",
            "mensagem": f"Erro ao descriptografar: {str(e)}",
            "texto": texto_criptografado
        }

def registra_acesso(usuario_id: str, documento_id: str, acao: str) -> Dict[str, Any]:
    """
    Registra um acesso ou ação em um documento.
    
    Args:
        usuario_id: ID do usuário
        documento_id: ID do documento
        acao: Tipo de ação (visualizar, editar, excluir, etc.)
        
    Returns:
        Registro de log
    """
    timestamp = datetime.datetime.now().isoformat()
    
    log = {
        "id": str(uuid.uuid4()),
        "timestamp": timestamp,
        "usuario_id": usuario_id,
        "documento_id": documento_id,
        "acao": acao,
        "ip": "127.0.0.1"  # Em uma implementação real, isso seria capturado
    }
    
    # Em uma implementação real, esse log seria armazenado em um banco de dados
    # Aqui apenas retornamos o log criado
    return log

def verifica_permissao(usuario_id: str, documento_id: str, permissoes: List[Dict[str, Any]]) -> bool:
    """
    Verifica se um usuário tem permissão para acessar um documento.
    
    Args:
        usuario_id: ID do usuário
        documento_id: ID do documento
        permissoes: Lista de permissões
        
    Returns:
        True se o usuário tem permissão, False caso contrário
    """
    # Verifica se existe uma permissão para o usuário e documento
    for permissao in permissoes:
        if (permissao.get("usuario_id") == usuario_id and 
            permissao.get("documento_id") == documento_id):
            return permissao.get("nivel_acesso", "nenhum") != "nenhum"
    
    return False

def aplica_politica_retencao(documento: Dict[str, Any], politica: str) -> Dict[str, Any]:
    """
    Aplica a política de retenção no documento.
    
    Args:
        documento: Dados do documento
        politica: Política de retenção (sessão, 7 dias, 30 dias, permanente)
        
    Returns:
        Documento com metadados de retenção atualizados
    """
    agora = datetime.datetime.now()
    
    # Define a data de expiração com base na política
    if politica == "sessão atual":
        # Expira no fim da sessão (simulado como 1 hora)
        expiracao = agora + datetime.timedelta(hours=1)
    elif politica == "7 dias":
        expiracao = agora + datetime.timedelta(days=7)
    elif politica == "30 dias":
        expiracao = agora + datetime.timedelta(days=30)
    elif politica == "permanente":
        expiracao = agora + datetime.timedelta(days=3650)  # ~10 anos
    else:
        # Padrão: 30 dias
        expiracao = agora + datetime.timedelta(days=30)
    
    # Atualiza os metadados do documento
    documento["metadados_retencao"] = {
        "politica": politica,
        "data_criacao": agora.isoformat(),
        "data_expiracao": expiracao.isoformat(),
        "status": "ativo"
    }
    
    return documento

def verifica_documentos_expirados(documentos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Verifica quais documentos estão expirados segundo a política de retenção.
    
    Args:
        documentos: Lista de documentos
        
    Returns:
        Lista de documentos expirados
    """
    agora = datetime.datetime.now()
    expirados = []
    
    for doc in documentos:
        metadados_retencao = doc.get("metadados_retencao", {})
        if metadados_retencao:
            data_expiracao_str = metadados_retencao.get("data_expiracao")
            if data_expiracao_str:
                try:
                    data_expiracao = datetime.datetime.fromisoformat(data_expiracao_str)
                    if data_expiracao <= agora:
                        expirados.append(doc)
                except (ValueError, TypeError):
                    # Se não conseguir interpretar a data, considera não expirado
                    pass
    
    return expirados

def remove_dados_sensíveis(texto: str, tipos_dados: List[str] = None) -> str:
    """
    Remove dados sensíveis específicos do texto.
    
    Args:
        texto: O texto a ser processado
        tipos_dados: Lista de tipos de dados a remover (cpf, email, telefone, etc.)
        
    Returns:
        Texto com dados sensíveis removidos
    """
    if tipos_dados is None:
        tipos_dados = ["cpf", "email", "telefone", "cartao_credito", "endereco"]
    
    # Padronização de tipos
    tipos_dados = [t.lower() for t in tipos_dados]
    
    # CPF e CNPJ
    if "cpf" in tipos_dados or "cnpj" in tipos_dados:
        texto = re.sub(r'\d{3}\.\d{3}\.\d{3}-\d{2}', '[CPF REMOVIDO]', texto)
        texto = re.sub(r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}', '[CNPJ REMOVIDO]', texto)
    
    # Email
    if "email" in tipos_dados:
        texto = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL REMOVIDO]', texto)
    
    # Telefone
    if "telefone" in tipos_dados:
        texto = re.sub(r'\(\d{2}\)\s*\d{4,5}-\d{4}', '[TELEFONE REMOVIDO]', texto)
        texto = re.sub(r'\d{4,5}-\d{4}', '[TELEFONE REMOVIDO]', texto)
    
    # Cartão de crédito
    if "cartao_credito" in tipos_dados:
        texto = re.sub(r'\d{4}\s*\d{4}\s*\d{4}\s*\d{4}', '[CARTÃO REMOVIDO]', texto)
        texto = re.sub(r'\d{4}\s\d{6}\s\d{5}', '[CARTÃO REMOVIDO]', texto)
    
    # Endereço
    if "endereco" in tipos_dados:
        # Simplificado para demonstração
        for palavra in ["rua", "avenida", "av.", "alameda", "praça", "travessa"]:
            padrao = fr'{palavra}\s+[^,;.]*\d+[^,;.]*', f'[ENDEREÇO REMOVIDO]'
            texto = re.sub(padrao[0], padrao[1], texto, flags=re.IGNORECASE)
    
    # CEP
    if "cep" in tipos_dados:
        texto = re.sub(r'\d{5}-\d{3}', '[CEP REMOVIDO]', texto)
    
    # RG
    if "rg" in tipos_dados:
        texto = re.sub(r'\d{1,2}\.\d{3}\.\d{3}-\d', '[RG REMOVIDO]', texto)
    
    return texto

def gera_token_acesso(usuario_id: str, duracao_horas: int = 24) -> Dict[str, Any]:
    """
    Gera um token de acesso temporário.
    
    Args:
        usuario_id: ID do usuário
        duracao_horas: Duração do token em horas
        
    Returns:
        Dicionário com token e informações
    """
    agora = datetime.datetime.now()
    expiracao = agora + datetime.timedelta(hours=duracao_horas)
    
    token_payload = {
        "usuario_id": usuario_id,
        "criado_em": agora.isoformat(),
        "expira_em": expiracao.isoformat(),
        "token_id": str(uuid.uuid4())
    }
    
    # Em uma implementação real, isso seria assinado com uma chave secreta
    token_string = hashlib.sha256(json.dumps(token_payload).encode()).hexdigest()
    
    return {
        "token": token_string,
        "payload": token_payload
    }

def valida_token(token: str, payload: Dict[str, Any]) -> bool:
    """
    Valida um token de acesso.
    
    Args:
        token: Token a ser validado
        payload: Dados do token
        
    Returns:
        True se o token for válido, False caso contrário
    """
    # Verifica expiração
    try:
        expiracao = datetime.datetime.fromisoformat(payload.get("expira_em", ""))
        if expiracao < datetime.datetime.now():
            return False
    except (ValueError, TypeError):
        return False
    
    # Em uma implementação real, verificaria a assinatura do token
    token_calculado = hashlib.sha256(json.dumps(payload).encode()).hexdigest()
    
    return token == token_calculado
