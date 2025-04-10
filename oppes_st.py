import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np



# função para selecionar a quantidade de linhas do dataframe
def mostra_qntd_linhas(dataframe):
    
    qntd_linhas = st.sidebar.slider('Selecione a quantidade de linhas que deseja mostrar na tabela', min_value = 1, max_value = len(dataframe), step = 1)

    #st.write(dataframe.head(qntd_linhas).style.format(subset = ['Idade'], formatter="{:.2f}"))
    st.write(dataframe.head(qntd_linhas).style.format(subset = ['Idade']))


st.title('Análise dos dados\n')
st.write('Nesse projeto vamos ')

# função que cria o gráfico
def plot_graf(dataframe, categoria):

    dados_plot = dataframe.query('Cidade == @categoria')

    fig, ax = plt.subplots(figsize=(8,6))
    ax = sns.scatterplot(x = 'Idade', y = 'Escala_1', hue='cor_da_pele', style='Sexo', data = dados_plot)
    ax.set_title(f'Média na escala 1 dos participantes da cidade de {categoria} por idade', fontsize = 16)
    ax.set_xlabel('Idade', fontsize = 12)
    ax.tick_params(rotation = 20, axis = 'x')
    ax.set_ylabel('Média na escala 1', fontsize = 12)
  
    return fig

# importando os dados
dados = pd.read_csv('bd_oppes2.csv')

st.title('Análise das tabelas\n')
st.write('Médias por escola')

# filtros para a tabela
opcao_1 = st.sidebar.checkbox('Mostrar tabela')
if opcao_1:

    st.sidebar.markdown('## Filtro para a tabela')

    categorias = list(dados['Cidade'].unique())
    categorias.append('Todas')

    categoria = st.sidebar.selectbox('Selecione a categoria para apresentar na tabela', options = categorias)

    if categoria != 'Todas':
        df_categoria = dados.query('Cidade == @categoria')
        mostra_qntd_linhas(df_categoria)      
    else:
        mostra_qntd_linhas(dados)

#filtro para o gráfico
st.sidebar.markdown('## Filtro para o gráfico')

categoria_grafico = st.sidebar.selectbox('Selecione a categoria para apresentar no gráfico', options = dados['Cidade'].unique())
figura = plot_graf(dados, categoria_grafico)
st.pyplot(figura)
