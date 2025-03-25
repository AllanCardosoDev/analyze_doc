# Analyse Doc

<div align="center">
  
![Analyse Doc Logo](https://via.placeholder.com/150x150.png?text=AD)

**Uma plataforma avançada de análise de documentos construída com IA**

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0%2B-green)](https://langchain.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

</div>

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Funcionalidades](#-funcionalidades)
- [Tecnologias Utilizadas](#-tecnologias-utilizadas)
- [Requisitos do Sistema](#-requisitos-do-sistema)
- [Instalação](#-instalação)
- [Guia de Uso](#-guia-de-uso)
- [Configuração Avançada](#-configuração-avançada)
- [Solução de Problemas](#-solução-de-problemas)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Desenvolvimento e Contribuições](#-desenvolvimento-e-contribuições)
- [Licença](#-licença)

## 🔍 Visão Geral

O Analyse Doc é uma aplicação poderosa para processamento e análise de documentos baseada em inteligência artificial. Desenvolvida com Streamlit e LangChain, a plataforma extrai, processa e analisa conteúdo de diversas fontes, permitindo que usuários obtenham insights valiosos por meio de uma interface conversacional intuitiva.

O sistema suporta múltiplos tipos de documentos, processamento avançado de texto e integração com os principais modelos de linguagem como Groq e OpenAI, tudo em uma interface amigável e responsiva.

## ✨ Funcionalidades

### Processamento de Documentos
- **Múltiplos Formatos**: Suporte nativo para PDF, DOCX, CSV, TXT e muito mais
- **Conteúdo Web**: Extração automática de conteúdo de sites e vídeos do YouTube
- **Processamento Inteligente**: Resumo automático e análise de texto

### Interface Conversacional
- **Chat Interativo**: Interaja com documentos através de perguntas em linguagem natural
- **Memória de Conversa**: O assistente mantém o contexto ao longo da conversa
- **Respostas em Tempo Real**: Visualize respostas sendo geradas em streaming

### Personalização e Controle
- **Seleção de Modelos**: Escolha entre diversos modelos de IA de diferentes provedores
- **Configurações Avançadas**: Ajuste parâmetros de processamento para suas necessidades
- **Tema Personalizável**: Alterne entre modos claro e escuro conforme sua preferência

### Recursos Avançados
- **Suporte a Proxy**: Contorne limitações e restrições de acesso a conteúdo
- **Processamento Multilíngue**: Trabalhe com documentos em diversos idiomas
- **Análise Detalhada**: Extraia insights estruturados de seus documentos

## 🛠 Tecnologias Utilizadas

| Categoria | Tecnologias |
|-----------|-------------|
| **Frontend** | Streamlit |
| **Backend** | Python, LangChain |
| **IA & ML** | Groq, OpenAI, NLTK, Scikit-learn |
| **Processamento de Dados** | Python-docx, PyPDF, BeautifulSoup |
| **Outros** | YouTube Transcript API, DuckDuckGo Search |

## 💻 Requisitos do Sistema

- Python 3.7 ou superior
- Acesso à Internet para modelos de IA em nuvem
- 4GB de RAM (mínimo), 8GB+ recomendado
- Chaves de API para Groq e/ou OpenAI

## 🚀 Instalação

### Instalação via Git

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/analyse-doc.git
cd analyse-doc

# Crie e ative um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt

# Configure as variáveis de ambiente
cp .env.example .env
# Edite o arquivo .env com suas chaves de API
```

### Execução do Aplicativo

```bash
streamlit run app.py
```

O aplicativo será iniciado e estará acessível em `http://localhost:8501` em seu navegador.

## 📝 Guia de Uso

### 1. Upload de Documentos

1. Na barra lateral, selecione a aba "Upload de Arquivos"
2. Escolha o tipo de documento que deseja analisar
3. Carregue o arquivo ou forneça a URL (para sites e vídeos)

### 2. Seleção de Modelo

1. Na aba "Seleção de Modelos", escolha o provedor (Groq ou OpenAI)
2. Selecione o modelo específico para sua análise
3. Insira sua chave de API no campo apropriado

### 3. Configurações de Processamento

1. Acesse a aba "Processamento" para opções avançadas
2. Ative recursos como resumo automático, se desejado
3. Ajuste configurações de idioma e outros parâmetros

### 4. Inicialização e Interação

1. Clique em "Inicializar Analyse Doc" para começar
2. Faça perguntas sobre o documento no chat
3. Receba respostas baseadas no conteúdo do documento

## ⚙️ Configuração Avançada

### Configuração de Proxy para YouTube

Se você encontrar bloqueios ao acessar conteúdo do YouTube:

1. Acesse a aba "Configurações"
2. Insira informações de proxy no formato: `http://usuario:senha@host:porta`
3. Reinicialize o Analyse Doc

### Personalização de Modelos

Para ajustar as configurações dos modelos de IA:

```python
# Exemplo de configuração personalizada (em app.py)
CONFIG_MODELOS = {
    "Groq": {
        "modelos": [
            "llama3-70b-8192",  # Adicione ou remova modelos conforme necessário
        ],
        "chat": ChatGroq,
        "parametros": {
            "temperature": 0.7,
            "max_tokens": 4000
        }
    }
}
```

## 🔧 Solução de Problemas

### Problemas com YouTube

**Problema**: Erro "IP blocked" ao tentar acessar vídeos do YouTube.

**Solução**: 
1. Configure um proxy como descrito na seção de configuração
2. Use uma VPN para alterar seu endereço IP
3. Aguarde algumas horas e tente novamente

### Erros de API

**Problema**: Falha na comunicação com APIs de modelos.

**Solução**:
1. Verifique se sua chave de API está correta
2. Confirme se sua conta tem créditos suficientes
3. Verifique a conectividade com a internet

### Problemas de Memória

**Problema**: Aplicativo lento o
