# Analyse Doc

Uma aplicação de chat para análise de documentos construída com [Streamlit](https://streamlit.io) que utiliza os modelos de linguagem da Groq e OpenAI para gerar respostas inteligentes sobre o conteúdo de documentos.

## Funcionalidades

- **Análise de Documentos**: Carregue diferentes tipos de arquivos e faça perguntas sobre o conteúdo
- **Suporte para Múltiplos Formatos**: PDF, DOCX, CSV, TXT e conteúdo da web (sites e vídeos do YouTube)
- **Modelos de IA Avançados**: Integração com Groq e OpenAI para respostas precisas
- **Resumo Automático de Documentos**: Gera automaticamente um resumo detalhado do documento carregado
- **Análise de Termos e Tópicos**: Extrai os principais tópicos e termos relevantes dos documentos
- **Exportação em PDF**: Baixe resumos formatados em PDF para compartilhamento
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
   streamlit run main.py
   ```

## Guia de Uso

1. Na barra lateral, selecione a aba "Upload de Arquivos"
2. Escolha o tipo de documento e faça o upload do arquivo ou forneça a URL
3. Na aba "Seleção de Modelos", escolha o provedor e o modelo de IA
4. Adicione sua chave de API
5. Configure opções de processamento na aba "Processamento" (resumo, idioma, etc.)
6. Clique em "Inicializar Analyse Doc"
7. Após a inicialização, o resumo automático do documento é gerado e pode ser visualizado
8. Use o chat para fazer perguntas específicas sobre o documento

## Resumo Automático de Documentos

A nova funcionalidade de resumo automático permite:

- Geração de um resumo detalhado assim que o documento é carregado
- Extração de tópicos principais e termos-chave do documento
- Opção de usar o modelo de IA para melhorar a qualidade do resumo
- Exportação do resumo completo em PDF com todas as análises
- Visualização do resumo a qualquer momento através do botão "Mostrar/Ocultar Resumo"

## Estrutura do Projeto

```
analyse-doc/
├── app.py                 # Aplicativo principal Streamlit
├── loaders.py             # Carregadores para diferentes tipos de documentos
├── resumo.py              # Módulo para geração de resumos automáticos
├── main.py                # Ponto de entrada principal
├── requirements.txt       # Dependências do projeto
└── .env                   # Arquivo de configuração (criar manualmente)
```

## Solução de Problemas

### Problemas com YouTube

Se você encontrar erros de "IP bloqueado" ao acessar vídeos do YouTube, tente:

1. Configurar um proxy na aba "Configurações"
2. Usar uma VPN
3. Esperar algumas horas e tentar novamente

### Problemas com Resumo Automático

Se o resumo automático não for gerado corretamente:

1. Verifique se todas as dependências estão instaladas (especialmente nltk e fpdf)
2. Tente usar um modelo de IA mais potente (como llama-3.3-70b-versatile ou gpt-4o)
3. Reduza o tamanho máximo do resumo se o documento for muito extenso

## Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo LICENSE para detalhes.
