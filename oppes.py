import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go

#%matplotlib inline

#função para selecionar a quantidade de linhas do dataframe
def mostra_qntd_linhas(dataframe):
    
    qntd_linhas = st.sidebar.slider('Selecione a quantidade de linhas que deseja mostrar na tabela', min_value = 1, max_value = len(dataframe), step = 1)

    #st.write(dataframe.head(qntd_linhas).style.format(subset = ['Idade'], formatter="{:.2f}"))
    st.write(dataframe.head(qntd_linhas).style.format(subset = ['Idade']))



st.title('OPPES\n')
st.header('Resultados do diagnóstico')

st.sidebar.title ('Navegação')
page = st.sidebar.selectbox('Select a page:',
['Geral', 'Escala 1', 'Escala 2', 'Escala 3', 'Escala 4', 'Escala 5',  'Escala 6', 'Escala 7', 'Escala 8', 'Escala 9', 'Escala 10', 'Escala 11'])




# -------------------Geral---------------------------------------------


if page == 'Geral':

	import pandas as pd
	import streamlit as st
	import numpy as np
	import plotly.express as px
	import plotly.graph_objects as go


	
	#-------------  Modelo geral (imagem)

	st.subheader('Modelo de relações entre as variáveis')

	st.image("modelo_teorico.png", caption="Modelo com as relações entre as variáveis, médias e desvio-padrão, assim como os valores dos coeficientes omega de cada escala. Os indicadores + e - ao lado de cada escala apontam a direção da influência; se +, contribui para aumentar o pertencimento, se - contribui para o não pertencimento à escola")


	st.divider()
#-------------  gráfico de barras das escalas, por cidade entre as escalas

	st.subheader('Não pertencimento à escola: média em cada escala, por cidade')

	chart_data = pd.read_csv('bd_oppes.csv')

	st.bar_chart(
    chart_data,
    x="cidade",
    y=['E1_ameacas'],
    #y=['E1_ameacas', 'Escala_2'],
    #color=["#FF0000", "#0000FF"],
    stack=False,
    horizontal=True
    ) 

	st.divider()

	st.subheader('Efeitos instituiçoes: média em cada escala, por cidade')

	chart_data = pd.read_csv('bd_oppes.csv')

	st.bar_chart(
    chart_data,
    x="cidade",
    y=['E2_situacoes_estresse', 'E3_4_agentes', 'E6_locais'],
    #y=['E1_ameacas', 'Escala_2'],
    #color=["#FF0000", "#0000FF"],
    stack=False,
    horizontal=True  

    )
    
	st.divider()
	st.subheader('Soma dos grupos de pertencimento: média em cada escala, por cidade')

	st.bar_chart(
    chart_data,
    x="cidade",
    y=['E7_soma_pertenca'],
    #y=['E1_ameacas', 'Escala_2'],
    #color=["#FF0000", "#0000FF"],
    stack=False,
    horizontal=True  

    )

	st.subheader('Efeito grupais: média em cada escala, por cidade')

	st.bar_chart(
    chart_data,
    x="cidade",
    y=['E8_qualid_relacoes', 'E9_trat_desig_grupos'],
    #y=['E1_ameacas', 'Escala_2'],
    #color=["#FF0000", "#0000FF"],
    stack=False,
    horizontal=True
    )  


	st.divider()
	st.subheader('Efeitos individuais: média em cada escala, por cidade')

	st.bar_chart(
    chart_data,
    x="cidade",
    y=['E5_disc_pessoal', 'E10_est_emoc_neg', 'E11_satisf_vida'],
    #y=['E1_ameacas', 'Escala_2'],
    #color=["#FF0000", "#0000FF"],
    stack=False,
    horizontal=True  

    )



    #-------------  diagrama de dispersão entre as escalas

	# Subtítulo
	st.subheader('Diagrama de dispersão entre as variáveis')

	# Carregar os dados
	dados = pd.read_csv('bd_oppes.csv')
	

	# Seleção das variáveis para os eixos X e Y
	x_axis = st.selectbox('Selecione a variável a ser incluída no eixo X', dados.columns[:-1])
	y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', dados.columns[:-1], index = 1)

	dados = dados.dropna(subset=[x_axis, y_axis])
	dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

	# Cálculo da linha de regressão
	m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)  # m é a inclinação e b é a interseção
	x_regressao = np.array(dados[x_axis])
	y_regressao = m * x_regressao + b  # Valores previstos

	# Criando o gráfico de dispersão
	fig = px.scatter(dados, x=x_axis, y=y_axis, title='Gráfico de Dispersão com Linha de Regressão.')

	# Adicionando a linha de regressão ao gráfico
	fig.add_trace(go.Scatter(x=x_regressao, y=y_regressao, mode='lines', name='Linha de Regressão', line=dict(color='red', width=3)))

	# Exibir o gráfico no Streamlit
	st.plotly_chart(fig)
    
    # Realizar a regressão linear
	X = dados[[x_axis]].values
	y = dados[y_axis].values
	model = LinearRegression()
	model.fit(X, y)
	# Obter coeficiente de regressão e intercepto
	coeficiente = model.coef_[0]
	intercepto = model.intercept_
	# Exibir resultados
	st.write(f'Coeficiente de regressão: {coeficiente:.3f}')
	st.write(f'Intercepto: {intercepto:.3f}')
	

	#st.divider()

	# função que cria o gráfico
	# def plot_graf(dataframe, categoria):

	#     dados_plot = dataframe.query('Cidade == @categoria')

	#     fig, ax = plt.subplots(figsize=(8,6))
	#     ax = sns.barplot(x = 'Cidade', y = 'Escala_1', data = dados_plot)
	#     ax.set_title(f'Média na escala 1 dos participantes da cidade de {categoria}', fontsize = 16)
	#     ax.set_xlabel('Cidade', fontsize = 12)
	#     ax.tick_params(rotation = 20, axis = 'x')
	#     ax.set_ylabel('Média na escala 1', fontsize = 12)
	  
	#     return fig

	# # importando os dados
	# dados = pd.read_csv('bd_oppes2.csv')

	# st.title('Análise das tabelas\n')
	# st.write('Médias por escola')

	# # filtros para a tabela
	# opcao_1 = st.sidebar.checkbox('Mostrar tabela')
	# if opcao_1:

	#     st.sidebar.markdown('## Filtro para a tabela')

	#     categorias = list(dados['Cidade'].unique())
	#     categorias.append('Todas')

	#     categoria = st.sidebar.selectbox('Selecione a categoria para apresentar na tabela', options = categorias)

	#     if categoria != 'Todas':
	#         df_categoria = dados.query('Cidade == @categoria')
	#         mostra_qntd_linhas(df_categoria)      
	#     else:
	#         mostra_qntd_linhas(dados)

	# #filtro para o gráfico
	# st.sidebar.markdown('## Filtro para o gráfico')

	# categoria_grafico = st.sidebar.selectbox('Selecione a categoria para apresentar no gráfico', options = dados['Cidade'].unique())

	# figura = plot_graf(dados, categoria_grafico)
	# st.pyplot(figura)

# ----------------------------Escala 1-----------------------------------------------

elif page == 'Escala 1':
	st.subheader('Ameaças nas escolas')
   
	def plot_graf(dataframe, categoria):

	    dados_plot = dataframe.query('Sexo == @categoria')

	    fig, ax = plt.subplots(figsize=(8,6))
	    ax = sns.barplot(x = 'cidade', y = 'E1_ameacas', hue="cor_da_pele", data = dados_plot)
	    ax.set_title(f'Média na escala 1 dos participantes do sexo {categoria} por cidade', fontsize = 16)
	    ax.set_xlabel('cidade', fontsize = 12)
	    ax.tick_params(rotation = 20, axis = 'x')
	    ax.set_ylabel('Média na escala 1', fontsize = 12)
	  
	    return fig

	# importando os dados
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.dropna(subset=['Sexo'])

	st.title('Escala 1\n')



	# #filtro para o gráfico
	st.sidebar.markdown('## Filtro para os resultados da Escala 1')

	categoria_grafico = st.sidebar.selectbox('Selecione o sexo do participante para apresentar no gráfico', options = dados['Sexo'].unique())
	figura = plot_graf(dados, categoria_grafico)
	st.pyplot(figura)

 

	st.divider()
	st.subheader('Diagrama de dispersão entre a variável idade, a escala selecionada e as variáveis cor da pele e o sexo do participante.')

	st.write('Use o filtro abaixo para selecionar as variáveis')
	
	

	dados = pd.read_csv('bd_oppes.csv')

	

	# Seleção das variáveis para os eixos X e Y
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['cidade'], axis=1)
	#dados = dados.drop(['cidade', 'série'], axis=1)

	# Seleção das variáveis para os eixos X e Y
	x_axis = ('Idade')
	y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', dados.columns[:-1], index = 8)

	dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
	dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')


	dados = dados.dropna(subset=[x_axis, y_axis])
	dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

	if not dados[x_axis].empty and not dados[y_axis].empty:
    		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
    # Use m e b aqui
	else:
    		st.warning("Não há dados suficientes para realizar a regressão linear.")

	# Cálculo da linha de regressão
	m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)  # m é a inclinação e b é a interseção
	x_regressao = np.array(dados[x_axis])
	y_regressao = m * x_regressao + b  # Valores previstos

	# Criando o gráfico de dispersão
	fig = px.scatter(dados, x=x_axis, y=y_axis, color='cor_da_pele', symbol = 'Sexo', title='Gráfico de Dispersão com Linha de Regressão.')
	#sns.scatterplot(x = 'Idade', y = 'Escala_1', hue='cor_da_pele', style='Sexo', data = dados_plot)

	# Adicionando a linha de regressão ao gráfico
	fig.add_trace(go.Scatter(x=x_regressao, y=y_regressao, mode='lines', name='Linha de Regressão', line=dict(color='red', width=3)))

	# Exibir o gráfico no Streamlit
	st.plotly_chart(fig)
    
    # Realizar a regressão linear
	X = dados[[x_axis]].values
	y = dados[y_axis].values
	model = LinearRegression()
	model.fit(X, y)
	# Obter coeficiente de regressão e intercepto
	coeficiente = model.coef_[0]
	intercepto = model.intercept_
	# Exibir resultados
	st.write(f'Coeficiente de regressão: {coeficiente:.3f}')
	st.write(f'Intercepto: {intercepto:.3f}')

	




# ----------------------------Escala 2-----------------------------------------------

elif page == 'Escala 2':
	st.subheader('Situações ameaçadoras nas escolas')
	dados = pd.read_csv('bd_oppes.csv')

	plt.style.use("ggplot")
	plt.figure(figsize = (20, 12))
	dados["E1_ameacas"].hist(bins = 40, ec = "k", alpha = .6, color = "royalblue")
	plt.title("Distribuição dos resultados da Escala 1")
	plt.xlabel("Valores")
	plt.ylabel("Contagem")

	indexes = dados[dados["E1_ameacas"] == 0].index
	dados = dados.drop(indexes)
	indexes = dados[dados["E1_ameacas"] == 0].index
	dados = dados.drop(indexes)


	notas_max = [dados["E1_ameacas"].max(), dados["E1_ameacas"].max(), dados["E1_ameacas"].max(), dados["E1_ameacas"].max()]
	notas_min = [dados["E1_ameacas"].min(), dados["E1_ameacas"].min(), dados["E1_ameacas"].min(), dados["E1_ameacas"].min()]

	plt.figure(figsize = (20, 14))
	bar_width = .35
	index = np.arange(4)

	plt.barh(index, 
	         notas_max, 
	         ec = "k", 
	         alpha = .6, 
	         color = "royalblue", 
	         height = bar_width, 
	         label = "Máxima")

	plt.barh(index - bar_width, 
	         notas_min, 
	         ec = "k", 
	         alpha = .6, 
	         color = "darkblue", 
	         height = bar_width, 
	         label = "Mínima")

	for i, v in enumerate(notas_max):
	    plt.text(v - 50, i, str(v))
	for i, v in enumerate(notas_min):
	    plt.text(v - 50, i - bar_width, str(v))
	        
	plt.yticks(index - bar_width / 2, ("E1_ameacas", "E2_situacoes_estresse", "E5_disc_pessoal", "E6_locais"))
	plt.title("Notas máximas e mínimas em cada escala")
	plt.legend()
	plt.show()



# ----------------------------Escala 3-----------------------------------------------


elif page == 'Escala 3':
	st.subheader('Atitudes dos professores')

	# função para selecionar a quantidade de linhas do dataframe
	def mostra_qntd_linhas(dataframe):
	    
	    qntd_linhas = st.sidebar.slider('Selecione a quantidade de linhas que deseja mostrar na tabela', min_value = 1, max_value = len(dataframe), step = 1)

	    st.write(dataframe.head(qntd_linhas).style.format(subset = ['Idade']))


	st.title('Análise dos dados\n')
	st.write('Nesse projeto vamos ')

	# função que cria o gráfico
	def plot_graf(dataframe, categoria):

	    dados_plot = dataframe.query('Sexo == @categoria')

	    fig, ax = plt.subplots(figsize=(8,6))
	    ax = sns.barplot(x = 'cidade', y = 'E1_ameacas', hue="cor_da_pele", data = dados_plot)
	    ax.set_title(f'Média na escala 1 dos participantes do sexo {categoria} por cidade', fontsize = 16)
	    ax.set_xlabel('cidade', fontsize = 12)
	    ax.tick_params(rotation = 20, axis = 'x')
	    ax.set_ylabel('Média na escala 1', fontsize = 12)
	  
	    return fig

	# importando os dados
	dados = pd.read_csv('bd_oppes.csv')

	st.title('Análise das tabelas\n')
	st.write('Médias por escola')

	# filtros para a tabela
	opcao_1 = st.sidebar.checkbox('Mostrar tabela')
	if opcao_1:

	    st.sidebar.markdown('## Filtro para a tabela')

	    categorias = list(dados['Sexo'].unique())
	    categorias.append('Todas')

	    categoria = st.sidebar.selectbox('Selecione a categoria para apresentar na tabela', options = categorias)

	    if categoria != 'Todas':
	        df_categoria = dados.query('Sexo == @categoria')
	        mostra_qntd_linhas(df_categoria)      
	    else:
	        mostra_qntd_linhas(dados)

	# #filtro para o gráfico
	st.sidebar.markdown('## Filtro para o gráfico')

	categoria_grafico = st.sidebar.selectbox('Selecione a categoria para apresentar no gráfico', options = dados['Sexo'].unique())
	figura = plot_graf(dados, categoria_grafico)
	st.pyplot(figura)

	# ameacas = sns.read_csv("bd_oppes2.csv")
	# sns.lmplot(data=acidentes, x='Idade', y='Escala_1')

# ----------------------------Escala 4-----------------------------------------------


elif page == 'Escala 4':
	st.subheader('Atitudes das instituições')

	dados = pd.read_csv('bd_oppes.csv')

	



# Create pair plot with custom settings

	#sns.pairplot(data=dados, hue="Sexo", diag_kind="kde", palette="husl")
	sns.pairplot(data=dados, diag_kind="kde", palette="husl")



# Set title

	plt.title("Pair")




# Show plot

	plt.show()


# ----------------------------Escala 5-----------------------------------------------
elif page == 'Escala 5':
	st.subheader('Situações pessoais ocorridas nas escolas')

# ----------------------------Escala 6-----------------------------------------------

elif page == 'Escala 6':
	st.subheader('Locais em que ocorreram os eventos nas escolas')


# ----------------------------Escala 7-----------------------------------------------


elif page == 'Escala 7':
	st.subheader('Grupos de pertença dos estudantes')



# ----------------------------Escala 8-----------------------------------------------


elif page == 'Escala 8':
	st.subheader('Interações ocorridas no ambiente escolar')

# ----------------------------Escala 9-----------------------------------------------

elif page == 'Escala 9':
	st.subheader('Situações que acontecem nas escolas')


# ----------------------------Escala 10-----------------------------------------------


elif page == 'Escala 10':
	st.subheader('Estados emocionais relatados pelos estudantes')


# ----------------------------Escala 11-----------------------------------------------


else: 
	st.subheader('Relatos dos estudantes sobre a satisfação com a vida')

