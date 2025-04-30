import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go


st.title('OPPES\n')
st.header('Resultados do diagnóstico')

st.sidebar.title ('Navegação')
page = st.sidebar.selectbox('Select a page:',
['Geral', 'Escala 1', 'Escala 2', 'Escala 3', 'Escala 4', 'Escala 5',  'Escala 6', 'Escala 7', 'Escala 8', 'Escala 9', 'Escala 10', 'Escala 11'])

@st.cache_data
def load_dados():
	dados = pd.read_csv('bd_oppes.csv')
	return dados

# usar o cache
dados = load_dados()

# limpar colunas sem uso
dados = dados.drop(['Concorda', 'DRs', 'etapa'], axis=1)

#st.write(dados)

# -------------------Geral---------------------------------------------
#-------------------Geral---------------------------------------------
#-------------------Geral-------------------------------------------------


if page == 'Geral':


	
	#-------------  Modelo geral (imagem)

	st.subheader('Modelo de relações entre as variáveis')

	if st.checkbox ("Marque aqui para visualizar o modelo teórico"):
		st.image("modelo_teorico.png", caption="Modelo com as relações entre as variáveis, médias e desvio-padrão, assim como os valores dos coeficientes omega de cada escala. Os indicadores + e - ao lado de cada escala apontam a direção da influência; se +, contribui para aumentar o pertencimento, se - contribui para o não pertencimento à escola")


	st.divider()
#-------------  médias e gráficos das escalas, por cidade

	st.subheader('Escala 1: Sentimento de não-pertença à escola')

	dados = load_dados()

	media = dados['E1_ameacas'].mean()
	st.write(f'Média Geral: {media:.3f}')

	# Calcular a avaliação média por cidade
	media_esc = dados.groupby("Cidade")[["E1_ameacas"]].mean().reset_index()

	# Criar o gráfico de barras para exibir a avaliação média
	fig_media_esc = px.bar(media_esc, y="E1_ameacas", x="Cidade",
	title="Escala 1 : Sentimento de não-pertença, por cidade")

	

	st.divider()

	dados = load_dados()


	
	dados_agrupados = dados.groupby('Cidade').agg({'E1_ameacas':'mean'})


	if st.checkbox ("Marque aqui para visualizar as médias, por cidade"):
		st.write(dados_agrupados)
	

	if st.checkbox ("Marque aqui para visualizar o gráfico das médias, por cidade"):
		st.plotly_chart(fig_media_esc, use_container_width=True)

	st.divider()

#-------------  Instituições


	st.subheader('Efeitos instituiçoes: média em cada escala, por cidade')

	dados = pd.read_csv('bd_oppes.csv')
	
	dados = load_dados()
	

# Calcular a média das variáveis por cidade
	dados_agrupados_inst = dados.groupby('Cidade')[['E2_situacoes_estresse', 'E3_4_agentes', 'E6_locais']].mean().round(3)

# Transformar os dados para o formato longo
	med_inst_long = pd.melt(dados_agrupados_inst.reset_index(), 
                       id_vars=['Cidade'], 
                       value_vars=['E2_situacoes_estresse', 'E3_4_agentes', 'E6_locais'], 
                       var_name='Tipo', 
                       value_name='Valor')
	media_geral_inst = dados[['E2_situacoes_estresse', 'E3_4_agentes', 'E6_locais']].mean()

	st.write(f'Média Geral: {media_geral_inst.iloc[0]:.3f}')

# Criar o gráfico de barras
	fig_media_inst = px.bar(med_inst_long, 
                        x="Cidade", 
                        y="Valor", 
                        color="Tipo",  # Diferenciar pelas diferentes variáveis
                        title="Média das Escalas 2, 3 e 6: Ameaças Institucionais: Professores, Escola e Locais",
                        labels={"Valor": "Média do Valor", "Cidade": "Cidade"})  # Ajustando os rótulos


	dados_agrupados_inst = dados.groupby('Cidade')[['E2_situacoes_estresse', 'E3_4_agentes', 'E6_locais']].mean().round(3)
	if st.checkbox ("Marque aqui para visualizar as médias, por cidade", key="med_inst"):
		st.write(dados_agrupados_inst)

	if st.checkbox ("Marque aqui para visualizar o gráfico das médias, por cidade", key="graf_inst"):
		st.plotly_chart(fig_media_inst, use_container_width=True)



	
	
#-------------  Grupos de pertença
	
    
	st.divider()
	st.subheader('Soma dos grupos de pertencimento: média em cada escala, por cidade')
	dados = pd.read_csv('bd_oppes.csv')
	dados = load_dados()
	
	media = dados['E7_soma_pertenca'].mean()
	st.write(f'Média: {media:.3f}')

	# Calcular a avaliação média por cidade
	media_pert = dados.groupby("Cidade")[["E7_soma_pertenca"]].mean().reset_index()

	# Criar o gráfico de barras para exibir a avaliação média
	fig_media_pert = px.bar(media_pert, y="E7_soma_pertenca", x="Cidade",
	title="Escala 1 : Média do número de grupos de pertencimento, por cidade")

	if st.checkbox ("Marque aqui para visualizar as médias, por cidade", key="med_pert"):
		st.write(media_pert)

	if st.checkbox ("Marque aqui para visualizar o gráfico das médias, por cidade", key="graf_pert"):
		st.plotly_chart(fig_media_pert)
	#st.write(dados_agrupados_inst)


#-------------  Efeitos grupais

	st.divider()

	st.subheader('Efeito grupais: média em cada escala, por cidade')

	dados = pd.read_csv('bd_oppes.csv')
	
	dados = load_dados()
	

# Calcular a média das variáveis por cidade
	dados_agrupados_grup = dados.groupby('Cidade')[['E8_qualid_relacoes', 'E9_trat_desig_grupos']].mean().round(3)

# Transformar os dados para o formato longo
	med_grup_long = pd.melt(dados_agrupados_grup.reset_index(), 
                       id_vars=['Cidade'], 
                       value_vars=['E8_qualid_relacoes', 'E9_trat_desig_grupos'], 
                       var_name='Tipo', 
                       value_name='Valor')

	
	media_grup = dados['E8_9_grupal'].mean()
	st.write(f'Média Geral: {media_grup:.3f}')

# Criar o gráfico de barras
	fig_media_grup = px.bar(med_grup_long, 
                        x="Cidade", 
                        y="Valor", 
                        color="Tipo",  # Diferenciar pelas diferentes variáveis
                        title="Média das Escalas 8 e 9: Ameaças grupais: Qualidade das relaçẽos intergrupais e tratamento desigual entre os grupos",
                        labels={"Valor": "Média do Valor", "Cidade": "Cidade"})  # Ajustando os rótulos


	dados_agrupados_grup = dados.groupby('Cidade')[['E8_qualid_relacoes', 'E9_trat_desig_grupos']].mean().round(3)
	if st.checkbox ("Marque aqui para visualizar as médias, por cidade", key="med_grup"):
		st.write(dados_agrupados_grup)

	if st.checkbox ("Marque aqui para visualizar o gráfico das médias, por cidade", key="graf_grup"):
		st.plotly_chart(fig_media_grup, use_container_width=True)
	


#-------------  Efeitos individuais

	st.divider()
	st.subheader('Efeitos individuais: média em cada escala, por cidade')

	dados = pd.read_csv('bd_oppes.csv')
	
	dados = load_dados()
	

# Calcular a média das variáveis por cidade
	dados_agrupados_ind = dados.groupby('Cidade')[['E5_disc_pessoal', 'E10_est_emoc_neg', 'E11_satisf_vida']].mean().round(3)

# Transformar os dados para o formato longo
	med_ind_long = pd.melt(dados_agrupados_ind.reset_index(), 
                       id_vars=['Cidade'], 
                       value_vars=['E5_disc_pessoal', 'E10_est_emoc_neg', 'E11_satisf_vida'], 
                       var_name='Tipo', 
                       value_name='Valor')

	media_ind = dados['E5_10_11_individ'].mean()
	st.write(f'Média Geral: {media_ind:.3f}')

# Criar o gráfico de barras
	fig_media_ind = px.bar(med_ind_long, 
                        x="Cidade", 
                        y="Valor", 
                        color="Tipo",  # Diferenciar pelas diferentes variáveis
                        title="Média das Escalas 5, 10 e 11: Ameaças Individuais: Ter pessoalmente sido objeto de discriminação, Estados emocionais negativos e Satisfação com a vida",
                        labels={"Valor": "Média do Valor", "Cidade": "Cidade"})  # Ajustando os rótulos


	dados_agrupados_ind = dados.groupby('Cidade')[['E5_disc_pessoal', 'E10_est_emoc_neg', 'E11_satisf_vida']].mean().round(3)
	if st.checkbox ("Marque aqui para visualizar as médias, por cidade", key="med_ind"):
		st.write(dados_agrupados_ind)

	if st.checkbox ("Marque aqui para visualizar o gráfico das médias, por cidade", key="graf_ind"):
		st.plotly_chart(fig_media_ind, use_container_width=True)

	st.divider()

 #-------------  diagrama de dispersão entre as escalas

	# Subtítulo
	st.subheader('Diagrama de dispersão entre as variáveis')

	# Carregar os dados
	dados = pd.read_csv('bd_oppes.csv')
	dados = load_dados()

	selec_escalas = dados[['E1_ameacas', 'E2_situacoes_estresse', 'E3_4_agentes', 'E5_disc_pessoal', 'E6_locais', 'E7_soma_pertenca', 'E8_qualid_relacoes', 'E9_trat_desig_grupos', 'E10_est_emoc_neg', 'E11_satisf_vida']]
	

	# Seleção das variáveis para os eixos X e Y
	x_axis = st.selectbox('Selecione a variável a ser incluída no eixo X', selec_escalas.columns[:-1], index = 1)
	y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', selec_escalas.columns[:-1], index = 0)

	selec_escalas = selec_escalas.dropna(subset=[x_axis, y_axis])
	selec_escalas = selec_escalas[np.isfinite(selec_escalas[x_axis]) & np.isfinite(selec_escalas[y_axis])]

	# Cálculo da linha de regressão
	m, b = np.polyfit(selec_escalas[x_axis], selec_escalas[y_axis], 1)  # m é a inclinação e b é a interseção
	x_regressao = np.array(selec_escalas[x_axis])
	y_regressao = m * x_regressao + b  # Valores previstos

	# Criando o gráfico de dispersão
	fig = px.scatter(selec_escalas, x=x_axis, y=y_axis, title='Gráfico de Dispersão com Linha de Regressão.')

	# Adicionando a linha de regressão ao gráfico
	fig.add_trace(go.Scatter(x=x_regressao, y=y_regressao, mode='lines', name='Linha de Regressão', line=dict(color='red', width=3)))

	# Exibir o gráfico no Streamlit
	st.plotly_chart(fig)
    
    # Realizar a regressão linear
	X = selec_escalas[[x_axis]].values
	y = selec_escalas[y_axis].values
	model = LinearRegression()
	model.fit(X, y)
	# Obter coeficiente de regressão e intercepto
	coeficiente = model.coef_[0]
	intercepto = model.intercept_
	# Exibir resultados
	st.write(f'Coeficiente de regressão: {coeficiente:.3f}')
	st.write(f'Intercepto: {intercepto:.3f}')
	

# ----------------------------Escala 1-----------------------------------------------
# ----------------------------Escala 1-----------------------------------------------
# ----------------------------Escala 1-----------------------------------------------

elif page == 'Escala 1':
	st.subheader('Ameaças nas escolas')

	dados = pd.read_csv('bd_oppes.csv')
	#dados = load_dados()

	med_e1 = dados['E1_ameacas'].mean()
	med_e1_1 = dados['Escala_1_1'].mean()
	med_e1_2 = dados['Escala_1_2'].mean()
	med_e1_3 = dados['Escala_1_3'].mean()
	med_e1_4 = dados['Escala_1_4'].mean()
	med_e1_5 = dados['Escala_1_5'].mean()
	med_e1_6 = dados['Escala_1_6'].mean()
	med_e1_7 = dados['Escala_1_7'].mean()
	med_e1_8 = dados['Escala_1_8'].mean()
	med_e1_9 = dados['Escala_1_9'].mean()

	st.write(f'Média Geral: {med_e1:.3f}')

	st.divider()
	st.subheader('Média de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 1", key="med_E1"):
		st.write(f'Média item 1: {med_e1_1:.3f}')
		st.write(f'Média item 2: {med_e1_2:.3f}')
		st.write(f'Média item 3: {med_e1_3:.3f}')
		st.write(f'Média item 4: {med_e1_4:.3f}')
		st.write(f'Média item 5: {med_e1_5:.3f}')
		st.write(f'Média item 6: {med_e1_6:.3f}')
		st.write(f'Média item 7: {med_e1_7:.3f}')
		st.write(f'Média item 8: {med_e1_8:.3f}')
		st.write(f'Média item 9: {med_e1_9:.3f}')

	st.divider()
	st.subheader('Média da escala, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E1_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E1_ameacas'].mean().reset_index()
	media_E1_por_cidade_sexo['Sexo_Cor'] = media_E1_por_cidade_sexo['Sexo'] + " - " + media_E1_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E1_por_cidade_sexo["Cidade"].unique()
	sexos = media_E1_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E1_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E1_por_cidade_sexo[
	    media_E1_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E1_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E1_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E1_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E1_ameacas'].mean().reset_index()
	media_E1_por_cidade_sexo['Sexo_Cor'] = media_E1_por_cidade_sexo['Sexo'] + " - " + media_E1_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E1_por_cidade_sexo["Cidade"].unique()
	sexos = media_E1_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E1_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E1_por_cidade_sexo[
	    media_E1_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E1_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E1_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E1_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E1_ameacas"].mean()

	        fig_media_E1 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E1_ameacas",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E1_ameacas"].round(2),
	            title="Escala 1: Média de Ameaças por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E1_ameacas": "Média de Ameaças",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E1.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 1: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E1.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média do Sentimento de não-pertencimento (Escala 1)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E1.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E1, use_container_width=True)

	        # 🔽 Filtros exibidos apenas após o gráfico
	        with st.expander("Ajuste os filtros abaixo", expanded=True):
	            nova_cidade = st.multiselect(
	                "Selecione a(s) cidade(s):",
	                cidades,
	                default=st.session_state["cidade_selecionada"],
	                key="cidade_final"
	            )
	            novo_sexo = st.multiselect(
	                "Selecione o(s) sexo(s):",
	                sexos,
	                default=st.session_state["sexo_selecionado"],
	                key="sexo_final"
	            )
	            nova_cor = st.multiselect(
	                "Selecione a(s) cor(es) da pele:",
	                cores_pele,
	                default=st.session_state["cor_selecionada"],
	                key="cor_final"
	            )

	            # Atualizar session_state e recarregar se necessário
	            filtros_modificados = False
	            if nova_cidade != st.session_state["cidade_selecionada"]:
	                st.session_state["cidade_selecionada"] = nova_cidade
	                filtros_modificados = True
	            if novo_sexo != st.session_state["sexo_selecionado"]:
	                st.session_state["sexo_selecionado"] = novo_sexo
	                filtros_modificados = True
	            if nova_cor != st.session_state["cor_selecionada"]:
	                st.session_state["cor_selecionada"] = nova_cor
	                filtros_modificados = True
	            if filtros_modificados:
	                st.rerun()

	st.divider()
	st.subheader('Diagrama de dispersão entre a variável idade, a escala selecionada e as variáveis cor da pele e o sexo do participante.')

	st.write('Use o filtro abaixo para selecionar as variáveis')
		
		

	dados = pd.read_csv('bd_oppes.csv')



	# Seleção das variáveis para os eixos X e Y
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)
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
	fig = px.scatter(dados, x=x_axis, y=y_axis, color='Cor_da_pele', symbol = 'Sexo', title='Gráfico de Dispersão com Linha de Regressão.')
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
# 	dados = pd.read_csv('bd_oppes.csv')

# 	plt.style.use("ggplot")
# 	plt.figure(figsize = (20, 12))
# 	dados["E1_ameacas"].hist(bins = 40, ec = "k", alpha = .6, color = "royalblue")
# 	plt.title("Distribuição dos resultados da Escala 1")
# 	plt.xlabel("Valores")
# 	plt.ylabel("Contagem")

# 	indexes = dados[dados["E1_ameacas"] == 0].index
# 	dados = dados.drop(indexes)
# 	indexes = dados[dados["E1_ameacas"] == 0].index
# 	dados = dados.drop(indexes)


# 	notas_max = [dados["E1_ameacas"].max(), dados["E1_ameacas"].max(), dados["E1_ameacas"].max(), dados["E1_ameacas"].max()]
# 	notas_min = [dados["E1_ameacas"].min(), dados["E1_ameacas"].min(), dados["E1_ameacas"].min(), dados["E1_ameacas"].min()]

# 	plt.figure(figsize = (20, 14))
# 	bar_width = .35
# 	index = np.arange(4)

# 	plt.barh(index, 
# 	         notas_max, 
# 	         ec = "k", 
# 	         alpha = .6, 
# 	         color = "royalblue", 
# 	         height = bar_width, 
# 	         label = "Máxima")

# 	plt.barh(index - bar_width, 
# 	         notas_min, 
# 	         ec = "k", 
# 	         alpha = .6, 
# 	         color = "darkblue", 
# 	         height = bar_width, 
# 	         label = "Mínima")

# 	for i, v in enumerate(notas_max):
# 	    plt.text(v - 50, i, str(v))
# 	for i, v in enumerate(notas_min):
# 	    plt.text(v - 50, i - bar_width, str(v))
	        
# 	plt.yticks(index - bar_width / 2, ("E1_ameacas", "E2_situacoes_estresse", "E5_disc_pessoal", "E6_locais"))
# 	plt.title("Notas máximas e mínimas em cada escala")
# 	plt.legend()
# 	#plt.show()



# # ----------------------------Escala 3-----------------------------------------------


elif page == 'Escala 3':
	st.subheader('Atitudes dos professores e da escola em relação aos conflitos')

# 	# função para selecionar a quantidade de linhas do dataframe
# 	# def mostra_qntd_linhas(dataframe):
	    
# 	#     qntd_linhas = st.sidebar.slider('Selecione a quantidade de linhas que deseja mostrar na tabela', min_value = 1, max_value = len(dataframe), step = 1)

# 	#     st.write(dataframe.head(qntd_linhas).style.format(subset = ['Idade']))


# 	# função que cria o gráfico
# 	def plot_graf(dataframe, categoria):

# 	    dados_plot = dataframe.query('Sexo == @categoria')

# 	    fig, ax = plt.subplots(figsize=(8,6))
# 	    ax = sns.barplot(x = 'cidade', y = 'E3_4_agentes', hue="cor_da_pele", data = dados_plot)
# 	    ax.set_title(f'Média nas escalas 2 e 4 dos participantes do sexo {categoria} por cidade e cor da pele', fontsize = 16)
# 	    ax.set_xlabel('cidade', fontsize = 12)
# 	    ax.tick_params(rotation = 20, axis = 'x')
# 	    ax.set_ylabel('Média nas escalas 3 e 4', fontsize = 12)
	  
# 	    return fig

# 	# importando os dados
# 	dados = pd.read_csv('bd_oppes.csv')


# 	# filtros para a tabela
# 	opcao_1 = st.sidebar.checkbox('Mostrar tabela')
# 	if opcao_1:

# 	    st.sidebar.markdown('## Filtro para a tabela')

# 	    categorias = list(dados['Sexo'].unique())
# 	    categorias.append('Todas')

# 	    categoria = st.sidebar.selectbox('Selecione a categoria para apresentar na tabela', options = categorias)

# 	    if categoria != 'Todas':
# 	        df_categoria = dados.query('Sexo == @categoria')
# 	        mostra_qntd_linhas(df_categoria)      
# 	    else:
# 	        mostra_qntd_linhas(dados)

# 	# #filtro para o gráfico
# 	st.sidebar.markdown('## Filtro para o gráfico')

# 	categoria_grafico = st.sidebar.selectbox('Selecione a categoria para apresentar no gráfico', options = dados['Sexo'].unique())
# 	figura = plot_graf(dados, categoria_grafico)
# 	st.pyplot(figura)

# 	# ameacas = sns.read_csv("bd_oppes2.csv")
# 	# sns.lmplot(data=acidentes, x='Idade', y='Escala_1')

# # ----------------------------Escala 4-----------------------------------------------


elif page == 'Escala 4':
	st.subheader('Atitudes das instituições')

# 	dados = pd.read_csv('bd_oppes.csv')

	



# # Create pair plot with custom settings

# 	#sns.pairplot(data=dados, hue="Sexo", diag_kind="kde", palette="husl")
# 	# sns.pairplot(data=dados, diag_kind="kde", palette="husl")



# # Set title

# 	# plt.title("Pair")




# # Show plot

# 	# plt.show()


# # ----------------------------Escala 5-----------------------------------------------
elif page == 'Escala 5':
 	st.subheader('Situações pessoais ocorridas nas escolas')

# # Cálculo de uma nova coluna 

# 	media_E1_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E1_ameacas'].mean().reset_index()
# 	media_E1_por_cidade_sexo['Sexo_Cor'] = media_E1_por_cidade_sexo['Sexo'] + " - " + media_E1_por_cidade_sexo['Cor_da_pele']

# # Nova coluna no gráfico
# 	fig_media_E1 = px.bar(
#     media_E1_por_cidade_sexo,
#     x="Cidade",
#     y="E1_ameacas",
#     color="Sexo_Cor",
#     barmode="group",
#     title="Escala 1: Média da escala 1 por cidade, sexo e cor da pele"
# )

# # Mostrar gráfico

# 	st.plotly_chart(fig_media_E1)



# # ----------------------------Escala 6-----------------------------------------------

elif page == 'Escala 6':
	st.subheader('Locais em que ocorreram os eventos nas escolas')


# # ----------------------------Escala 7-----------------------------------------------


elif page == 'Escala 7':
	st.subheader('Grupos de pertença dos estudantes')



# # ----------------------------Escala 8-----------------------------------------------


elif page == 'Escala 8':
	st.subheader('Interações ocorridas no ambiente escolar')

# # ----------------------------Escala 9-----------------------------------------------

elif page == 'Escala 9':
	st.subheader('Situações que acontecem nas escolas')


# # ----------------------------Escala 10-----------------------------------------------


elif page == 'Escala 10':
	st.subheader('Estados emocionais relatados pelos estudantes')


# # ----------------------------Escala 11-----------------------------------------------


else: 
	st.subheader('Relatos dos estudantes sobre a satisfação com a vida')

