# Analyse Doc

Uma aplicação de chat para análise de documentos construída com [Streamlit](https://streamlit.io) que utiliza os modelos de linguagem da Groq e OpenAI para gerar respostas inteligentes sobre o conteúdo de documentos.

## Funcionalidades

- **Análise de Documentos**: Carregue diferentes tipos de arquivos e faça perguntas sobre o conteúdo
- **Suporte para Múltiplos Formatos**: PDF, DOCX, CSV, TXT e conteúdo da web (sites e vídeos do YouTube)
- **Modelos de IA Avançados**: Integração com Groq e OpenAI para respostas precisas
- **Processamento Básico**: Resumo automático e outras opções de processamento
- **Interface Amigável**: Chat intuitivo para interagir com seus documentos

## Como Executar Localmente

### Pré-requisitos

- Python 3.7 ou superior
- Chave de API da Groq ou OpenAI

### Instalação

1. **Clone o Repositório**

   ```bash
   git clone https://github.com/seu-usuario/analyse-doc.git
   cd analyse-doc
   ```

2. **Instale as Dependências**

   ```bash
   pip install -r requirements.txt
   ```

3. **Configure as Chaves de API**

   Crie um arquivo `.env` na raiz do projeto e adicione suas chaves:

   ```
   GROQ_API_KEY=sua_chave_aqui
   OPENAI_API_KEY=sua_chave_aqui
   ```

4. **Execute o Aplicativo**

   ```bash
   streamlit run app.py
   ```

## Guia de Uso

1. Na barra lateral, selecione a aba "Upload de Arquivos"
2. Escolha o tipo de documento e faça o upload do arquivo ou forneça a URL
3. Na aba "Seleção de Modelos", escolha o provedor e o modelo de IA
4. Adicione sua chave de API
5. Opcionalmente, configure opções de processamento na aba respectiva
6. Clique em "Inicializar Analyse Doc"
7. Use o chat para fazer perguntas sobre o documento

## Solução de Problemas

### Problemas com YouTube

Se você encontrar erros de "IP bloqueado" ao acessar vídeos do YouTube, tente:

1. Configurar um proxy na aba "Configurações"
2. Usar uma VPN
3. Esperar algumas horas e tentar novamente

## Estrutura do Projeto

```
analyse-doc/
├── app.py                 # Aplicativo principal
├── loaders.py             # Carregadores para diferentes tipos de documentos
├── requirements.txt       # Dependências do projeto
└── .env                   # Arquivo de configuração (criar manualmente)
```

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para detalhes.
