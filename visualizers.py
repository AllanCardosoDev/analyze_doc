"""
Módulo de visualizadores para o Analyse Doc.
Contém funções para geração de gráficos, visualizações e elementos visuais
para análise de documentos.
"""

import io
import re
import base64
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from datetime import datetime
import streamlit as st

# Tentar importar wordcloud, mas é opcional
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

def gera_nuvem_palavras(texto, max_palavras=100, cores=None):
    """
    Gera uma nuvem de palavras a partir do texto.
    
    Args:
        texto: O texto para gerar a nuvem
        max_palavras: Número máximo de palavras
        cores: Cores para a nuvem (primária, secundária)
        
    Returns:
        Imagem da nuvem de palavras em formato bytes
    """
    if not WORDCLOUD_AVAILABLE:
        # Se wordcloud não estiver disponível, retorna um erro visual
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "WordCloud não disponível\nInstale a biblioteca wordcloud", 
                ha='center', va='center', fontsize=20)
        ax.axis('off')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf
    
    # Stopwords em português
    stopwords = set([
        'a', 'ao', 'aos', 'aquela', 'aquelas', 'aquele', 'aqueles', 'aquilo', 'as', 'até',
        'com', 'como', 'da', 'das', 'de', 'dela', 'delas', 'dele', 'deles', 'depois',
        'do', 'dos', 'e', 'ela', 'elas', 'ele', 'eles', 'em', 'entre', 'era', 'eram',
        'essa', 'essas', 'esse', 'esses', 'esta', 'estas', 'este', 'estes',
        'eu', 'foi', 'foram', 'há', 'isso', 'isto', 'já', 'lhe', 'lhes', 'mais',
        'mas', 'me', 'mesmo', 'meu', 'meus', 'minha', 'minhas', 'muito', 'na', 'não', 
        'nas', 'nem', 'no', 'nos', 'nós', 'nossa', 'nossas', 'nosso', 'nossos', 'num', 
        'numa', 'o', 'os', 'ou', 'para', 'pela', 'pelas', 'pelo', 'pelos', 'por', 
        'qual', 'quando', 'que', 'quem', 'são', 'se', 'seja', 'sejam', 'sem', 'seu', 
        'seus', 'só', 'sua', 'suas', 'também', 'te', 'tem', 'um', 'uma', 'você', 'vocês'
    ])
    
    # Configuração de cores
    if cores is None:
        colormap = "viridis"
    else:
        # Cria um colormap personalizado com as cores fornecidas
        from matplotlib.colors import LinearSegmentedColormap
        cor_primaria = cores.get("primaria", "#1E88E5")
        cor_secundaria = cores.get("secundaria", "#FF5722")
        colormap = LinearSegmentedColormap.from_list("custom", [cor_primaria, cor_secundaria])
    
    # Configurações da nuvem
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        max_words=max_palavras,
        stopwords=stopwords,
        background_color='white',
        colormap=colormap,
        contour_width=1,
        contour_color='steelblue'
    )
    
    # Gera a nuvem
    wordcloud.generate(texto)
    
    # Salva em um buffer de bytes
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def gera_grafico_entidades(entidades, tipo="barras", cores=None):
    """
    Gera um gráfico das entidades extraídas.
    
    Args:
        entidades: Lista de entidades
        tipo: Tipo de gráfico (barras, pizza, etc.)
        cores: Cores para o gráfico
        
    Returns:
        Figura matplotlib
    """
    if not entidades:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Nenhuma entidade encontrada", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Configuração de estilo e cores
    if cores:
        cor_primaria = cores.get("primaria", "#1E88E5")
        cor_secundaria = cores.get("secundaria", "#FF5722")
        paleta = [cor_primaria, cor_secundaria] * 10  # Alternância de cores
    else:
        paleta = sns.color_palette("viridis", 10)
    
    # Conta a frequência das entidades
    contador = Counter(entidades)
    top_entidades = contador.most_common(10)  # Top 10 entidades
    
    labels = [item[0] for item in top_entidades]
    values = [item[1] for item in top_entidades]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if tipo == "barras":
        ax.bar(range(len(labels)), values, color=paleta)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Frequência')
        ax.set_title('Entidades mais frequentes')
        plt.tight_layout()
    
    elif tipo == "pizza":
        wedges, texts, autotexts = ax.pie(
            values, labels=labels, autopct='%1.1f%%',
            colors=paleta, startangle=90
        )
        ax.axis('equal')
        ax.set_title('Distribuição de entidades')
        plt.setp(autotexts, size=8, weight="bold")
    
    elif tipo == "horizontal":
        ax.barh(range(len(labels)), values, color=paleta)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel('Frequência')
        ax.set_title('Entidades mais frequentes')
        plt.tight_layout()
    
    return fig

def gera_mapa_conexoes(texto, max_nodes=20):
    """
    Gera um mapa de conexões entre conceitos e entidades no texto.
    
    Args:
        texto: O texto para analisar
        max_nodes: Número máximo de nós no grafo
        
    Returns:
        Gráfico plotly interativo
    """
    # Esta é uma implementação simplificada para fins de demonstração
    # Uma versão real usaria NLP para extrair relacionamentos entre entidades
    
    # Extraímos todos os substantivos e adjetivos como "conceitos"
    words = re.findall(r'\b\w{4,15}\b', texto.lower())
    word_counts = Counter(words)
    
    # Pegamos os conceitos mais comuns como nós
    top_words = [w for w, c in word_counts.most_common(max_nodes)]
    
    # Criamos um grafo não direcionado
    G = nx.Graph()
    
    # Adicionamos os nós
    for word in top_words:
        G.add_node(word, size=word_counts[word])
    
    # Adicionamos arestas baseadas na co-ocorrência de palavras
    window_size = 5  # Palavras que ocorrem dentro dessa janela são consideradas conectadas
    
    words_in_text = [w.lower() for w in re.findall(r'\b\w{4,15}\b', texto)]
    
    for i, word in enumerate(words_in_text):
        if word in top_words:
            # Encontra palavras ao redor dela
            start = max(0, i - window_size)
            end = min(len(words_in_text), i + window_size)
            
            for j in range(start, end):
                if i != j and words_in_text[j] in top_words:
                    G.add_edge(word, words_in_text[j])
    
    # Cria o layout do grafo
    pos = nx.spring_layout(G, seed=42)
    
    # Preparamos dados para plotly
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_size.append(G.nodes[node]['size'] * 2)  # Tamanho proporcional à frequência
    
    # Criamos o gráfico interativo com plotly
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            colorbar=dict(
                thickness=15,
                title='Frequência',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Mapa de Conexões de Conceitos',
                       titlefont_size=16,
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )
    
    return fig

def gera_timeline(texto):
    """
    Gera uma timeline com eventos mencionados no texto.
    
    Args:
        texto: O texto para analisar
        
    Returns:
        Gráfico plotly com timeline
    """
    # Esta função extrairia datas e eventos associados
    # Implementação simplificada para demonstração
    
    # Procura por padrões de data no texto seguidos por contexto
    datas_encontradas = []
    
    # Padrão: dia/mês/ano
    padrao_data1 = r'(\d{1,2}/\d{1,2}/\d{4})(.*?)(?:\.|;|$)'
    matches = re.findall(padrao_data1, texto)
    for match in matches:
        data_str, contexto = match
        try:
            # Converte para timestamp
            data = datetime.strptime(data_str, "%d/%m/%Y")
            datas_encontradas.append({
                "data": data,
                "evento": contexto.strip()[:100],  # Limita tamanho do contexto
                "formato": "DD/MM/AAAA"
            })
        except:
            pass
    
    # Padrão: dia de mês de ano
    meses = {
        'janeiro': 1, 'fevereiro': 2, 'março': 3, 'abril': 4, 'maio': 5, 'junho': 6,
        'julho': 7, 'agosto': 8, 'setembro': 9, 'outubro': 10, 'novembro': 11, 'dezembro': 12
    }
    
    padrao_data2 = r'(\d{1,2})\s+de\s+(janeiro|fevereiro|março|abril|maio|junho|julho|agosto|setembro|outubro|novembro|dezembro)\s+de\s+(\d{4})(.*?)(?:\.|;|$)'
    matches = re.findall(padrao_data2, texto, re.IGNORECASE)
    
    for match in matches:
        dia, mes, ano, contexto = match
        try:
            # Converte para timestamp
            mes_num = meses[mes.lower()]
            data = datetime(int(ano), mes_num, int(dia))
            datas_encontradas.append({
                "data": data,
                "evento": contexto.strip()[:100],
                "formato": "textual"
            })
        except:
            pass
    
    # Se não encontrou datas suficientes, gera algumas para demonstração
    if len(datas_encontradas) < 3:
        # Cria uma timeline artificial para demonstração
        base_date = datetime(2023, 1, 1)
        datas_encontradas = [
            {"data": datetime(2023, 1, 1), "evento": "Início do documento", "formato": "exemplo"},
            {"data": datetime(2023, 3, 15), "evento": "Desenvolvimento", "formato": "exemplo"},
            {"data": datetime(2023, 6, 30), "evento": "Conclusão", "formato": "exemplo"}
        ]
    
    # Ordena as datas
    datas_encontradas.sort(key=lambda x: x["data"])
    
    # Cria o dataframe para o gráfico
    df = pd.DataFrame([{
        "Data": data["data"],
        "Evento": data["evento"],
        "Tipo": data["formato"]
    } for data in datas_encontradas])
    
    # Cria o gráfico
    fig = px.timeline(df, x_start="Data", y="Evento", color="Tipo",
                     title="Timeline de Eventos")
    
    # Personaliza o layout
    fig.update_layout(
        xaxis=dict(
            title="Data",
            type="date"
        ),
        yaxis=dict(
            title=None
        )
    )
    
    return fig

def gera_grafico_estatisticas(dados, tipo="barras", titulo=None, cores=None):
    """
    Gera um gráfico de estatísticas baseado nos dados fornecidos.
    
    Args:
        dados: Dicionário ou DataFrame com dados
        tipo: Tipo de gráfico (barras, linha, dispersão)
        titulo: Título do gráfico
        cores: Cores para o gráfico
        
    Returns:
        Figura matplotlib
    """
    if not dados:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Sem dados para visualização", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Configuração de cores
    if cores:
        cor_primaria = cores.get("primaria", "#1E88E5")
        cor_secundaria = cores.get("secundaria", "#FF5722")
    else:
        cor_primaria = "#1E88E5"
        cor_secundaria = "#FF5722"
    
    # Converte para DataFrame se for dicionário
    if isinstance(dados, dict):
        df = pd.DataFrame(dados)
    else:
        df = dados
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if tipo == "barras":
        if len(df.columns) > 1:
            df.plot(kind='bar', ax=ax, color=[cor_primaria, cor_secundaria])
        else:
            df.plot(kind='bar', ax=ax, color=cor_primaria)
    
    elif tipo == "linha":
        if len(df.columns) > 1:
            df.plot(kind='line', ax=ax, marker='o', color=[cor_primaria, cor_secundaria])
        else:
            df.plot(kind='line', ax=ax, marker='o', color=cor_primaria)
    
    elif tipo == "dispersao":
        if len(df.columns) >= 2:
            ax.scatter(df.iloc[:, 0], df.iloc[:, 1], color=cor_primaria)
            ax.set_xlabel(df.columns[0])
            ax.set_ylabel(df.columns[1])
    
    elif tipo == "pizza":
        df.iloc[0].plot(kind='pie', ax=ax, autopct='%1.1f%%', colors=[cor_primaria, cor_secundaria])
    
    # Adiciona título
    if titulo:
        ax.set_title(titulo)
    
    plt.tight_layout()
    return fig

def gera_heatmap(matriz, colunas=None, linhas=None, titulo=None, cores=None):
    """
    Gera um heatmap a partir de uma matriz de dados.
    
    Args:
        matriz: Matriz de dados (lista de listas ou numpy array)
        colunas: Rótulos das colunas
        linhas: Rótulos das linhas
        titulo: Título do gráfico
        cores: Mapa de cores
        
    Returns:
        Figura matplotlib
    """
    # Converte para array numpy se necessário
    matriz_np = np.array(matriz)
    
    # Define cores
    if cores:
        cmap = cores
    else:
        cmap = "viridis"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Cria o heatmap
    heatmap = sns.heatmap(matriz_np, annot=True, fmt=".2f", cmap=cmap, ax=ax,
                         xticklabels=colunas, yticklabels=linhas)
    
    # Adiciona título
    if titulo:
        ax.set_title(titulo)
    
    plt.tight_layout()
    return fig

def gera_grafico_sentimento(sentimentos, titulo="Análise de Sentimento", cores=None):
    """
    Gera um gráfico de análise de sentimento ao longo do texto.
    
    Args:
        sentimentos: Lista de valores de sentimento
        titulo: Título do gráfico
        cores: Cores para o gráfico
        
    Returns:
        Figura matplotlib
    """
    if not sentimentos:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, "Sem dados de sentimento", ha='center', va='center', fontsize=14)
        ax.axis('off')
        return fig
    
    # Configuração de cores
    if cores:
        cor_positivo = cores.get("positivo", "#4CAF50")
        cor_negativo = cores.get("negativo", "#F44336")
        cor_neutro = cores.get("neutro", "#9E9E9E")
    else:
        cor_positivo = "#4CAF50"  # Verde
        cor_negativo = "#F44336"  # Vermelho
        cor_neutro = "#9E9E9E"    # Cinza
    
    # Cria o gráfico
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Eixo X: posição no texto (ou trecho)
    x = range(len(sentimentos))
    
    # Determina cores baseadas no valor do sentimento
    cores_pontos = [cor_positivo if s > 0.1 else cor_negativo if s < -0.1 else cor_neutro for s in sentimentos]
    
    # Plota o gráfico de linha
    ax.plot(x, sentimentos, color='#2196F3', alpha=0.7)
    
    # Adiciona pontos coloridos
    ax.scatter(x, sentimentos, c=cores_pontos, s=50, zorder=5)
    
    # Adiciona linha horizontal no zero (neutro)
    ax.axhline(y=0, color='#9E9E9E', linestyle='-', alpha=0.5)
    
    # Adiciona áreas coloridas para positivo/negativo
    ax.fill_between(x, sentimentos, 0, where=(np.array(sentimentos) > 0), 
                   color=cor_positivo, alpha=0.2)
    ax.fill_between(x, sentimentos, 0, where=(np.array(sentimentos) < 0), 
                   color=cor_negativo, alpha=0.2)
    
    # Define os limites e rótulos
    ax.set_ylim(-1.1, 1.1)
    ax.set_ylabel('Sentimento')
    ax.set_xlabel('Posição no texto')
    ax.set_title(titulo)
    
    # Adiciona rótulos para as áreas
    plt.text(len(sentimentos)/2, 0.8, 'Positivo', ha='center', color=cor_positivo)
    plt.text(len(sentimentos)/2, -0.8, 'Negativo', ha='center', color=cor_negativo)
    
    plt.tight_layout()
    return fig
