import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
import pingouin as pg


st.image('logo_oppes.png')
st.header('Resultados do diagnóstico')

st.sidebar.title ('Navegação')
page = st.sidebar.selectbox('Selecione uma página:',
['Geral', 'Escala 1', 'Escala 2', 'Escalas 3 e 4', 'Escala 5',  'Escala 6', 'Escala 7', 'Escala 8', 'Escala 9', 'Escala 10', 'Escala 11', 'Modelos de regressão', 'Dados Textuais'])

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

	descricoes = {
	    'E1_ameacas': 'Sentimento de não-pertencimento ao ambiente escolar.',
	    'E2_situacoes_estresse': 'Número de situações estressantes enfrentadas.',
	    'E3_4_agentes': 'Ação dos professores e da escola.',
	    'E5_disc_pessoal': 'Qualidade da disciplina pessoal observada.',
	    'E6_locais': 'Avaliação da segurança e conforto dos locais da escola.',
	    'E7_soma_pertenca': 'Soma dos grupos de pertencimento.',
	    'E8_qualid_relacoes': 'Qualidade das relações entre os grupos.',
	    'E9_trat_desig_grupos': 'Tratamento desigual entre os grupos.',
	    'E10_est_emoc_neg': 'Estados emocionais negativos.',
	    'E11_satisf_vida': 'Satisfação com a vida.'

	}

	

	if st.checkbox ("Marque aqui para visualizar o modelo teórico"):
		st.image("modelo_teorico.png", caption="Modelo com as relações entre as variáveis, médias e desvio-padrão, assim como os valores dos coeficientes omega de cada escala. Os indicadores + e - ao lado de cada escala apontam a direção da influência; se +, contribui para aumentar o pertencimento, se - contribui para o não pertencimento à escola")
		st.markdown("**Descrição das variáveis independentes:**")
		for variavel, descricao in descricoes.items():
			st.write(f'**{variavel}**: {descricao}')


	st.divider()
#-------------  médias e gráficos das escalas, por cidade

	st.subheader('Escala 1: Sentimento de não-pertença à escola')

	dados = load_dados()

	media = dados['E1_ameacas'].mean()
	st.markdown(f'**Média geral: {media:.3f}**')

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

	st.markdown(f'**Média Geral: {media_geral_inst.iloc[0]:.3f}**')


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
	
	media_pert = dados['E7_soma_pertenca'].mean()
	st.markdown(f'**Média geral: {media_pert:.3f}**')

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
	st.markdown(f'**Média geral: {media_grup:.3f}**')

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
	st.markdown(f'**Média geral: {media_ind:.3f}**')

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

#-------- ANOVAS
	st.markdown(f"## ANOVAs (Comparar diferenças das médias de cada escala entre as cidades)")


	st.divider()
	df = pd.read_csv("bd_oppes.csv")

	# Definir as variáveis
	variaveis = ['E1_ameacas', 'E2_situacoes_estresse', 'E3_4_agentes', 'E5_disc_pessoal',
	             'E6_locais', 'E7_soma_pertenca', 'E8_qualid_relacoes', 'E9_trat_desig_grupos',
	             'E10_est_emoc_neg', 'E11_satisf_vida']

	# Loop para análise
	for var in variaveis:
	    with st.expander(f"🔍 Resultados - {var}", expanded=False):
	        
	        # Remover valores ausentes
	        dados_validos = df[[var, 'Cidade']].dropna()

	        # Checkbox e resultado da ANOVA
	        if st.checkbox(f"Mostrar ANOVA - {var}", value=False):
	            model = ols(f'{var} ~ C(Cidade)', data=dados_validos).fit()
	            anova_table = sm.stats.anova_lm(model, typ=2)
	            st.markdown("**Tabela ANOVA**")
	            st.dataframe(anova_table)

	        # Checkbox e resultado do teste de Tukey
	        if st.checkbox(f"Mostrar Teste de Tukey - {var}", value=False):
	            mc = MultiComparison(dados_validos[var], dados_validos['Cidade'])
	            tukey_result = mc.tukeyhsd()
	            tukey_df = pd.DataFrame(data=tukey_result._results_table.data[1:], 
	                                    columns=tukey_result._results_table.data[0])
	            st.markdown("**Teste Post Hoc (Tukey HSD)**")
	            st.dataframe(tukey_df)

	        # Checkbox e gráfico do teste de Tukey
	        if st.checkbox(f"Mostrar Gráfico de Tukey - {var}", value=False):
	            mc = MultiComparison(dados_validos[var], dados_validos['Cidade'])
	            tukey_result = mc.tukeyhsd()
	            fig = tukey_result.plot_simultaneous()
	            st.pyplot(fig)

	        # Checkbox e resultado de Games-Howell
	        if st.checkbox(f"Mostrar Games-Howell - {var}", value=False):
	            gh_result = pg.pairwise_gameshowell(dv=var, between='Cidade', data=dados_validos)
	            st.markdown("**Teste Post Hoc (Games-Howell)**")
	            st.dataframe(gh_result)

 #-------------  diagrama de dispersão entre as escalas

	# Subtítulo
	

	st.subheader('Diagrama de dispersão e regressão linear simples: Dados Brutos vs. Padronizados')

	# Carregamento de dados
	@st.cache_data
	def load_dados():
	    return pd.read_csv("bd_oppes.csv")

	dados = load_dados()

	# Colunas numéricas de interesse
	colunas_escalas = [
	    'E1_ameacas', 'E2_situacoes_estresse', 'E3_4_agentes',
	    'E5_disc_pessoal', 'E6_locais', 'E7_soma_pertenca',
	    'E8_qualid_relacoes', 'E9_trat_desig_grupos',
	    'E10_est_emoc_neg', 'E11_satisf_vida'
	]

	# Função para remover outliers com IQR
	def remover_outliers_iqr(df, colunas):
	    Q1 = df[colunas].quantile(0.25)
	    Q3 = df[colunas].quantile(0.75)
	    IQR = Q3 - Q1
	    filtro = ~((df[colunas] < (Q1 - 1.5 * IQR)) | (df[colunas] > (Q3 + 1.5 * IQR))).any(axis=1)
	    return df[filtro]

	# Limpeza e remoção de outliers
	dados_limpos = dados[colunas_escalas].dropna()
	dados_filtrados = remover_outliers_iqr(dados_limpos, colunas_escalas)

	# Padronização
	scaler = StandardScaler()
	dados_padronizados = pd.DataFrame(
	    scaler.fit_transform(dados_filtrados),
	    columns=colunas_escalas
	)

	# Seleção das variáveis
	x_var = st.selectbox("Selecione a variável para o eixo X", colunas_escalas, index=1)
	y_var = st.selectbox("Selecione a variável para o eixo Y", colunas_escalas, index=0)

	# Função para regressão linear
	def realizar_regressao(X, y):
	    modelo = LinearRegression().fit(X, y)
	    y_pred = modelo.predict(X)
	    return modelo, y_pred

	# ===== GRÁFICO 1: Dados Brutos (Sem Remoção de Outliers) =====
	X_bruto = dados_limpos[[x_var]].values
	y_bruto = dados_limpos[y_var].values
	modelo_bruto, y_pred_bruto = realizar_regressao(X_bruto, y_bruto)

	fig_bruto = px.scatter(dados_limpos, x=x_var, y=y_var, title="Gráfico de dispersão com os dados brutos (Sem Remoção de Outliers)")
	fig_bruto.add_trace(go.Scatter(x=dados_limpos[x_var], y=y_pred_bruto, mode="lines", name="Regressão", line=dict(color="red", width=2)))

	# ===== GRÁFICO 2: Dados Com Outliers Removidos =====
	X_filtrado = dados_filtrados[[x_var]].values
	y_filtrado = dados_filtrados[y_var].values
	modelo_filtrado, y_pred_filtrado = realizar_regressao(X_filtrado, y_filtrado)

	fig_filtrado = px.scatter(dados_filtrados, x=x_var, y=y_var, title="Gráfico de dispersão com os dados brutos e outliers removidos")
	fig_filtrado.add_trace(go.Scatter(x=dados_filtrados[x_var], y=y_pred_filtrado, mode="lines", name="Regressão", line=dict(color="red", width=2)))

	# ===== GRÁFICO 3: Dados Padronizados (Sem Remoção de Outliers) =====
	X_padronizados = scaler.fit_transform(dados_limpos[[x_var]])  # Apenas padronizando X
	y_padronizados = dados_limpos[y_var].values  # Usando y original, sem padronizar

	modelo_padronizados, y_pred_padronizados = realizar_regressao(X_padronizados, y_padronizados)

	fig_padronizados = px.scatter(dados_limpos, x=x_var, y=y_var, title="Dados Padronizados (Sem Remoção de Outliers)")
	fig_padronizados.add_trace(go.Scatter(x=dados_limpos[x_var], y=y_pred_padronizados, mode="lines", name="Regressão", line=dict(color="red", width=2)))

	# ===== GRÁFICO 4: Dados Padronizados Com Outliers Removidos =====
	X_pad_filtrado = scaler.fit_transform(dados_filtrados[[x_var]])  # Apenas padronizando X
	y_pad_filtrado = dados_filtrados[y_var].values  # Usando y original, sem padronizar

	modelo_pad_filtrado, y_pred_pad_filtrado = realizar_regressao(X_pad_filtrado, y_pad_filtrado)

	fig_pad_filtrado = px.scatter(dados_filtrados, x=x_var, y=y_var, title="Dados Padronizados com Outliers Removidos")
	fig_pad_filtrado.add_trace(go.Scatter(x=dados_filtrados[x_var], y=y_pred_pad_filtrado, mode="lines", name="Regressão", line=dict(color="red", width=2)))

	# ===== MOSTRAR GRÁFICOS LADO A LADO =====
	col1, col2 = st.columns(2)
	if st.checkbox ("Marque aqui para visualizar os gráficos com os dados brutos, com e sem outliers", key="graficos"):
		with col1:
		    st.plotly_chart(fig_bruto, use_container_width=True)

		with col2:
		    st.plotly_chart(fig_filtrado, use_container_width=True)
		    

	# ===== MOSTRAR GRÁFICOS PADRONIZADOS LADO A LADO =====
	
	col3, col4 = st.columns(2)
	if st.checkbox ("Marque aqui para visualizar os resultados, com dados brutos e padronizados, com e sem outliers ", key="graf_reg_p"):
		with col3:
			st.markdown("**Regressão - Dados Brutos (Sem Remoção de Outliers)**")
			st.write(f"Coeficiente: {modelo_bruto.coef_[0]:.3f}")
			st.write(f"Intercepto: {modelo_bruto.intercept_:.3f}")
			st.write(f"R²: {modelo_bruto.score(X_bruto, y_bruto):.3f}")
			st.markdown("**Regressão - Dados com Outliers Removidos**")
			st.write(f"Coeficiente: {modelo_filtrado.coef_[0]:.3f}")
			st.write(f"Intercepto: {modelo_filtrado.intercept_:.3f}")
			st.write(f"R²: {modelo_filtrado.score(X_filtrado, y_filtrado):.3f}")
		    

		with col4:
			st.markdown("**Regressão - Dados Padronizados (Sem Outliers)**")
			st.write(f"Coeficiente: {modelo_padronizados.coef_[0].item():.3f}")
			st.write(f"Intercepto: {modelo_padronizados.intercept_.item():.3f}")
			st.write(f"R²: {modelo_padronizados.score(X_padronizados, y_padronizados):.3f}")
			st.markdown("**Regressão - Dados Padronizados com Outliers Removidos**")
			st.write(f"Coeficiente: {modelo_pad_filtrado.coef_[0].item():.3f}")
			st.write(f"Intercepto: {modelo_pad_filtrado.intercept_.item():.3f}")
			st.write(f"R²: {modelo_pad_filtrado.score(X_pad_filtrado, y_pad_filtrado):.3f}")

	


	

# ----------------------------Escala 1-----------------------------------------------
# ----------------------------Escala 1-----------------------------------------------
# ----------------------------Escala 1-----------------------------------------------

elif page == 'Escala 1':
	st.subheader('Sentimento de não-pertencimento ao ambiente escolar')

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

	st.markdown(f'**Média geral: {med_e1:.3f}**')

	st.divider()
	st.subheader('Média de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 1", key="med_E1"):
		st.write(f'1 = {med_e1_1:.3f} (Na minha escola há muitas brigas e desentendimentos entre os alunos(as)).')
		st.write(f'2 = {med_e1_2:.3f} (Estou satisfeito com a relação que tenho com meus colegas, na minha escola.)')
		st.write(f'3 = {med_e1_3:.3f} (A minha escola é um lugar onde me sinto excluído(a).)')
		st.write(f'4 = {med_e1_4:.3f} (A minha escola é um lugar onde faço amigos (as) com facilidade.)')
		st.write(f'5 = {med_e1_5:.3f} (Sinto que faço parte da minha escola.)')
		st.write(f'6 = {med_e1_6:.3f} (A minha escola é um lugar onde os outros gostam de mim.)')
		st.write(f'7 = {med_e1_7:.3f} (A minha escola é um lugar onde me sinto só.)')
		st.write(f'8 = {med_e1_8:.3f} (Durante as aulas, expresso as minhas opiniões.)')
		st.write(f'9 = {med_e1_9:.3f} (Se eu pudesse, eu mudaria de escola.)')

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
	

	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E1 = [
	    'E1_ameacas', 'Escala_1_1', 'Escala_1_2',
	    'Escala_1_3', 'Escala_1_4', 'Escala_1_5',
	    'Escala_1_6', 'Escala_1_7', 
	    'Escala_1_8', 'Escala_1_9'
	]
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc2"):
		# === Interface de seleção ===
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E1, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
	

		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')

	




# ----------------------------Escala 2-----------------------------------------------
# ----------------------------Escala 2-----------------------------------------------
# ----------------------------Escala 2-----------------------------------------------

elif page == 'Escala 2':
	st.subheader('Situações estressantes nas escolas')

	med_e2 = dados['E2_situacoes_estresse'].mean()
	med_e2_1 = dados['Escala_2_1'].mean()
	med_e2_2 = dados['Escala_2_2'].mean()
	med_e2_3 = dados['Escala_2_3'].mean()
	med_e2_4 = dados['Escala_2_4'].mean()
	med_e2_5 = dados['Escala_2_5'].mean()
	med_e2_6 = dados['Escala_2_6'].mean()
	med_e2_7 = dados['Escala_2_7'].mean()
	med_e2_8 = dados['Escala_2_8'].mean()
	med_e2_9 = dados['Escala_2_9'].mean()
	med_e2_10 = dados['Escala_2_10'].mean()
	med_e2_11 = dados['Escala_2_11'].mean()
	med_e2_12 = dados['Escala_2_12'].mean()
	med_e2_13 = dados['Escala_2_13'].mean()
	med_e2_14 = dados['Escala_2_14'].mean()
	med_e2_15 = dados['Escala_2_15'].mean()
	med_e2_16 = dados['Escala_2_16'].mean()
	
	st.markdown(f'**Média geral: {med_e2:.3f}**')

	st.divider()
	st.subheader('Média de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 1", key="med_E1"):
		st.write(f'1 = {med_e2_1:.3f} (Os(as) alunos(as) desrespeitam os(as) professores(as).)')
		st.write(f'2 = {med_e2_2:.3f} (Os(as) funcionários tratam todos os (as)alunos(as) com respeito.)')
		st.write(f'3 = {med_e2_3:.3f} (0s (as) alunos(as) ofendem ou ameaçam alguns professores.)')
		st.write(f'4 = {med_e2_4:.3f} (Os(as) professores escutam o que os alunos têm a dizer.)')
		st.write(f'5 = {med_e2_5:.3f} (Os(as) professores implicam com alguns alunos.)')
		st.write(f'6 = {med_e2_6:.3f} (Os(as) professores ameaçam alguns alunos.)')
		st.write(f'7 = {med_e2_7:.3f} (Os(as) professores tiram sarro ou humilham alguns alunos.)')
		st.write(f'8 = {med_e2_8:.3f} (Os(as) professores conversam com os alunos sobre problemas de convivência.)')
		st.write(f'9 = {med_e2_9:.3f} (Alguns alunos(as) vêm para a escola após usar bebidas alcoólicas ou outras drogas.)')
		st.write(f'10 = {med_e2_10:.3f} (Alguns alunos (as) vendem drogas dentro da escola.)')
		st.write(f'11 = {med_e2_11:.3f} (Alguns alunos(as) trazem facas, canivetes, estiletes etc., como armas para a escola.)')
		st.write(f'12 = {med_e2_12:.3f} (Os(as) professores têm alunos favoritos.)')
		st.write(f'13 = {med_e2_13:.3f} (Na maior parte das vezes, a escola pune ou dá bronca no grupo todo e não apenas nos envolvidos nas confusões.)')
		st.write(f'14 = {med_e2_14:.3f} (As punições são impostas sem o(a) aluno(a) ser ouvido(a).)')
		st.write(f'15 = {med_e2_15:.3f} (Quando ocorrem situações de conflitos podemos contar com os professores e a direção da escola para ajudar a resolvê-los.)')
		st.write(f'16 = {med_e2_16:.3f} (Os conflitos na escola são resolvidos de forma justa para os envolvidos.)')
	

	st.divider()
	st.subheader('Média da escala, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E2_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E2_situacoes_estresse'].mean().reset_index()
	media_E2_por_cidade_sexo['Sexo_Cor'] = media_E2_por_cidade_sexo['Sexo'] + " - " + media_E2_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E2_por_cidade_sexo["Cidade"].unique()
	sexos = media_E2_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E2_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E2_por_cidade_sexo[
	    media_E2_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E2_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E2_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E2_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E2_situacoes_estresse'].mean().reset_index()
	media_E2_por_cidade_sexo['Sexo_Cor'] = media_E2_por_cidade_sexo['Sexo'] + " - " + media_E2_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E2_por_cidade_sexo["Cidade"].unique()
	sexos = media_E2_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E2_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E2_por_cidade_sexo[
	    media_E2_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E2_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E2_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E2_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E2_situacoes_estresse"].mean()

	        fig_media_E2 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E2_situacoes_estresse",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E2_situacoes_estresse"].round(2),
	            title="Escala 2: Média de Situações de estresse por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E2_situacoes_estresse": "Média de Situações de estresse",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E2.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 2: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E2.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média de situações de estresse (Escala 2)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E2.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E2, use_container_width=True)

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


	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E2 = [
	    'E2_situacoes_estresse', 'Escala_2_1', 'Escala_2_2',
	    'Escala_2_3', 'Escala_2_4', 'Escala_2_5',
	    'Escala_2_6', 'Escala_2_7', 
	    'Escala_2_8', 'Escala_2_9', 'Escala_2_10', 'Escala_2_11', 'Escala_2_12',
	    'Escala_2_13', 'Escala_2_14', 
	    'Escala_2_15', 'Escala_2_16', 'Escala_2_17', 'Escala_2_18'
	]

	# === Interface de seleção ===
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc2"):
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E2, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')


# # ----------------------------Escalas 3 e 4-----------------------------------------------
# # ----------------------------Escalas 3 e 4-----------------------------------------------
# # ----------------------------Escalas 3 e 4-----------------------------------------------


elif page == 'Escalas 3 e 4':
	st.subheader('Atitudes dos professores e da escola em relação aos conflitos')

	med_e3_4 = dados['E3_4_agentes'].mean()
	med_e3_1 = dados['Escala_3_1'].mean()
	med_e3_2 = dados['Escala_3_2'].mean()
	med_e3_3 = dados['Escala_3_3'].mean()
	med_e3_4 = dados['Escala_3_4'].mean()
	med_e3_5 = dados['Escala_3_5'].mean()
	med_e3_6 = dados['Escala_3_6'].mean()
	med_e3_7 = dados['Escala_3_7'].mean()
	med_e3_8 = dados['Escala_3_8'].mean()

	med_e4_1 = dados['Escala_4_1'].mean()
	med_e4_2 = dados['Escala_4_2'].mean()
	med_e4_3 = dados['Escala_4_3'].mean()
	med_e4_4 = dados['Escala_4_4'].mean()
	med_e4_5 = dados['Escala_4_5'].mean()
	med_e4_6 = dados['Escala_4_6'].mean()
	med_e4_7 = dados['Escala_4_7'].mean()
	med_e4_8 = dados['Escala_4_8'].mean()
	med_e4_9 = dados['Escala_4_9'].mean()
	
	
	st.markdown(f'**Média geral: {med_e3_4:.3f}**')

	st.divider()
	st.subheader('Média de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 1", key="med_E1"):
		st.markdown('**Como os professores reagem aos conflitos**')
		st.write(f'1 = {med_e3_1:.3f} (Fingem que não perceberam)')
		st.write(f'2 = {med_e3_2:.3f} (Informam a família sobre o ocorrido para que tome providências)')
		st.write(f'3 = {med_e3_3:.3f} (Os(as) alunos(as) envolvidos são ouvidos e incentivados a buscar soluções para o problema)')
		st.write(f'4 = {med_e3_4:.3f} (Colocam os(as) alunos(as) para fora da sala de aula)')
		st.write(f'5 = {med_e3_5:.3f} (Mudam os(as) alunos(as) de lugar na sala de aula)')
		st.write(f'6 = {med_e3_6:.3f} (Retiram um objeto (celular, fone de ouvido, etc.) que pertence ao(a) aluno(a))')
		st.write(f'7 = {med_e3_7:.3f} (Não sabem o que fazer)')
		st.write(f'8 = {med_e3_8:.3f} (Encaminham para a direção/coordenação/orientação)')

		st.markdown('**Como a escola reage aos conflitos**')
		st.write(f'1 = {med_e4_1:.3f} (Os(as) alunos(as) envolvidos recebem advertência oralmente ou por escrito)')
		st.write(f'2 = {med_e4_2:.3f} (Os alunos envolvidos são humilhados na frente dos colegas)')
		st.write(f'3 = {med_e4_3:.3f} (Os alunos envolvidos são suspensos)')
		st.write(f'4 = {med_e4_4:.3f} (Os alunos envolvidos são ouvidos e convidados a reparar seus erros)')
		st.write(f'5 = {med_e4_5:.3f} (A escola informa o ocorrido à família, pedindo que tome providências)')
		st.write(f'6 = {med_e4_6:.3f} (Os(as) alunos(as) são encaminhados(as) para o Conselho Tutelar)')
		st.write(f'7 = {med_e4_7:.3f} (A escola não sabe o que fazer)')
		st.write(f'8 = {med_e4_8:.3f} (A escola registra um boletim de ocorrência na polícia)')
		st.write(f'9 = {med_e4_9:.3f} (A escola impede que alunos (as) participem de atividades que gostam (recreio, Educação Física, festa, excursão etc.))')
	

	st.divider()
	st.subheader('Média da escala, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E3_4_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E3_4_agentes'].mean().reset_index()
	media_E3_4_por_cidade_sexo['Sexo_Cor'] = media_E3_4_por_cidade_sexo['Sexo'] + " - " + media_E3_4_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E3_4_por_cidade_sexo["Cidade"].unique()
	sexos = media_E3_4_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E3_4_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E3_4_por_cidade_sexo[
	    media_E3_4_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E3_4_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E3_4_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E3_4_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E3_4_agentes'].mean().reset_index()
	media_E3_4_por_cidade_sexo['Sexo_Cor'] = media_E3_4_por_cidade_sexo['Sexo'] + " - " + media_E3_4_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E3_4_por_cidade_sexo["Cidade"].unique()
	sexos = media_E3_4_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E3_4_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E3_4_por_cidade_sexo[
	    media_E3_4_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E3_4_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E3_4_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E3_4_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E3_4_agentes"].mean()

	        fig_media_E3_4 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E3_4_agentes",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E3_4_agentes"].round(2),
	            title="Escalas 3 e 4: Influência dos professores e da escola, por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E3_4_agentes": "impacto dos professores e da escola (Escalas 3 e 4)",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E3_4.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 3: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E3_4.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média do impacto dos professores e da escola (Escalas 3 e 4)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E3_4.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E3_4, use_container_width=True)

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


	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E3_4 = [
	    'E3_4_agentes', 'Escala_3_1', 'Escala_3_2',
	    'Escala_3_3', 'Escala_3_4', 'Escala_3_5',
	    'Escala_3_6', 'Escala_3_7', 
	    'Escala_3_8', 'Escala_4_1', 'Escala_4_2', 'Escala_4_3', 'Escala_4_4',
	    'Escala_4_5', 'Escala_4_6', 
	    'Escala_4_7', 'Escala_4_8', 'Escala_4_9'
	]

	# === Interface de seleção ===
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc3_4"):
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E3_4, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')



# # ----------------------------Escala 5-----------------------------------------------
# # ----------------------------Escala 5-----------------------------------------------
# # ----------------------------Escala 5-----------------------------------------------
elif page == 'Escala 5':
	st.subheader('Situações de discriminação pessoal ocorridas nas escolas')
	dados = pd.read_csv('bd_oppes.csv')
	med_e5 = dados['E5_disc_pessoal'].mean()
	med_e5_1 = dados['Escala_6_1'].mean()
	med_e5_2 = dados['Escala_6_2'].mean()
	med_e5_3 = dados['Escala_6_3'].mean()
	med_e5_4 = dados['Escala_6_4'].mean()
	med_e5_5 = dados['Escala_6_5'].mean()
	med_e5_6 = dados['Escala_6_6'].mean()
	med_e5_7 = dados['Escala_6_7'].mean()
	

	st.markdown(f'**Média geral: {med_e5:.3f}**')

	st.divider()
	st.subheader('Média de cada item da escala 5')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 5", key="med_E5"):
		st.write(f'1 = {med_e5_1:.3f} (Eu fui agredida(o), maltratada(o), intimidada(o), ameaçada(o), excluída(o) ou humilhada(o) por algum(a) colega da escola ou professor)')
		st.write(f'2 = {med_e5_2:.3f} (Eu fui provocada(o), zoada(o), apelidada(o) ou irritada(o) por algum(a) colega da escola ou professor)')
		st.write(f'3 = {med_e5_3:.3f} (Eu tenho medo de alguns alunos)')
		st.write(f'4 = {med_e5_4:.3f} (Eu agredi, maltratei, intimidei, ameacei, exclui ou humilhei algum(a) colega ou professor(a) da escola)')
		st.write(f'5 = {med_e5_5:.3f} (Eu provoquei, zoei, coloquei apelidos ou irritei algum(a) colega ou professor(a) da escola)')
		st.write(f'6 = {med_e5_6:.3f} (Eu vi alguém sendo agredida(o), maltratada(o), intimidada(o), ameaçada(o), excluída(o) ou humilhada(o) por algum(a) colega ou professor(a) da escola)')
		st.write(f'7 = {med_e5_7:.3f} (Eu vi alguém sendo provocada(o), zoada(o), recebendo apelidos ouirritada(o) por algum(a) colega ou professor(a) da escola)')
		

	st.divider()
	st.subheader('Média da escala, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E5_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E5_disc_pessoal'].mean().reset_index()
	media_E5_por_cidade_sexo['Sexo_Cor'] = media_E5_por_cidade_sexo['Sexo'] + " - " + media_E5_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E5_por_cidade_sexo["Cidade"].unique()
	sexos = media_E5_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E5_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E5_por_cidade_sexo[
	    media_E5_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E5_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E5_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E5_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E5_disc_pessoal'].mean().reset_index()
	media_E5_por_cidade_sexo['Sexo_Cor'] = media_E5_por_cidade_sexo['Sexo'] + " - " + media_E5_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E5_por_cidade_sexo["Cidade"].unique()
	sexos = media_E5_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E5_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E5_por_cidade_sexo[
	    media_E5_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E5_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E5_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E1_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E5_disc_pessoal"].mean()

	        fig_media_E5 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E5_disc_pessoal",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E5_disc_pessoal"].round(2),
	            title="Escala 1: Média de ter sofrido discriminação pessoal, por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E5_disc_pessoal": "Média de de ter sofrido discriminação pessoal",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E5.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 5: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E5.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média do Sentimento de de ter sofrido discriminação pessoal (Escala 5)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E5.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E5, use_container_width=True)

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
	

	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E5 = [
	    'E1_ameacas', 'Escala_6_1', 'Escala_6_2',
	    'Escala_6_3', 'Escala_6_4', 'Escala_6_6',
	    'Escala_6_6', 'Escala_6_7'
	]
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc5"):
		# === Interface de seleção ===
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E5, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
	

		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')



# # ----------------------------Escala 6-----------------------------------------------
# # ----------------------------Escala 6-----------------------------------------------
# # ----------------------------Escala 6-----------------------------------------------

elif page == 'Escala 6':
	st.subheader('Locais em que ocorreram os eventos nas escolas')

	dados = pd.read_csv('bd_oppes.csv')
	med_e6 = dados['E6_locais'].mean()
	med_e6_1 = dados['Escala_6_1'].mean()
	med_e6_2 = dados['Escala_6_2'].mean()
	med_e6_3 = dados['Escala_6_3'].mean()
	med_e6_4 = dados['Escala_6_4'].mean()
	med_e6_5 = dados['Escala_6_5'].mean()
	med_e6_6 = dados['Escala_6_6'].mean()
	med_e6_7 = dados['Escala_6_7'].mean()
	med_e6_8 = dados['Escala_6_8'].mean()

	st.markdown(f'**Média geral: {med_e6:.3f}**')

	st.divider()
	st.subheader('Média de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 6", key="med_E6"):
		st.write(f'1 = {med_e6_1:.3f} (Na classe)')
		st.write(f'2 = {med_e6_2:.3f} (Nos corredores)')
		st.write(f'3 = {med_e6_3:.3f} (No pátio)')
		st.write(f'4 = {med_e6_4:.3f} (No refeitório/cantina)')
		st.write(f'5 = {med_e6_5:.3f} (Nos banheiros)')
		st.write(f'6 = {med_e6_6:.3f} (Na quadra)')
		st.write(f'7 = {med_e6_7:.3f} (Em locais próximos à escola)')
		st.write(f'8 = {med_e6_8:.3f} (Através da internet ou celular)')

	st.divider()
	st.subheader('Média da escala de locais em que sofre discriminação, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E6_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E6_locais'].mean().reset_index()
	media_E6_por_cidade_sexo['Sexo_Cor'] = media_E6_por_cidade_sexo['Sexo'] + " - " + media_E6_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E6_por_cidade_sexo["Cidade"].unique()
	sexos = media_E6_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E6_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E6_por_cidade_sexo[
	    media_E6_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E6_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E6_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E6_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E6_locais'].mean().reset_index()
	media_E6_por_cidade_sexo['Sexo_Cor'] = media_E6_por_cidade_sexo['Sexo'] + " - " + media_E6_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E6_por_cidade_sexo["Cidade"].unique()
	sexos = media_E6_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E6_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E6_por_cidade_sexo[
	    media_E6_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E6_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E6_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E1_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E6_locais"].mean()

	        fig_media_E6 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E6_locais",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E6_locais"].round(2),
	            title="Escala 6: Média dos locais em que sofreu discriminação pessoal, por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E6_locais": "Média dos locais em que sofreu discriminação pessoal",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E6.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 6: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E6.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média dos locais em que sofreu discriminação pessoal (Escala 6)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E6.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E6, use_container_width=True)

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
	

	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E6 = [
	    'E6_locais', 'Escala_6_1', 'Escala_6_2',
	    'Escala_6_3', 'Escala_6_4', 'Escala_6_5',
	    'Escala_6_6', 'Escala_6_7', 'Escala_6_8'
	]
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc6"):
		# === Interface de seleção ===
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E6, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
	

		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')




# # ----------------------------Escala 7-----------------------------------------------
# # ----------------------------Escala 7-----------------------------------------------
# # ----------------------------Escala 7-----------------------------------------------


elif page == 'Escala 7':
	st.subheader('Grupos de pertença dos estudantes')

	dados = pd.read_csv('bd_oppes.csv')
	med_e7 = dados['E7_soma_pertenca'].mean()
	med_e7_1 = dados['Escala_7_1'].mean()
	med_e7_2 = dados['Escala_7_2'].mean()
	med_e7_3 = dados['Escala_7_3'].mean()
	med_e7_4 = dados['Escala_7_4'].mean()
	med_e7_5 = dados['Escala_7_5'].mean()
	med_e7_6 = dados['Escala_7_6'].mean()
	med_e7_7 = dados['Escala_7_7'].mean()
	med_e7_8 = dados['Escala_7_8'].mean()
	med_e7_9 = dados['Escala_7_9'].mean()
	med_e7_10 = dados['Escala_7_10'].mean()
	med_e7_11 = dados['Escala_7_11'].mean()
	med_e7_12 = dados['Escala_7_12'].mean()
	med_e7_13 = dados['Escala_7_13'].mean()
	med_e7_14 = dados['Escala_7_14'].mean()
	med_e7_15 = dados['Escala_7_15'].mean()
	med_e7_16 = dados['Escala_7_16'].mean()
	med_e7_17 = dados['Escala_7_17'].mean()
	med_e7_18 = dados['Escala_7_18'].mean()
	med_e7_19 = dados['Escala_7_19'].mean()
	med_e7_20 = dados['Escala_7_20'].mean()
	med_e7_21 = dados['Escala_7_21'].mean()
	med_e7_22 = dados['Escala_7_22'].mean()


	st.markdown(f'**Média geral: {med_e7:.3f}**')

	st.divider()
	st.subheader('Proporção de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a proporção em cada item da Escala 7", key="med_E7"):
		st.write(f'Homem: {med_e7_1:.3f}')
		st.write(f'Mulher: {med_e7_2:.3f}')
		st.write(f'Cisgênero: {med_e7_3:.3f}')
		st.write(f'Transgênero: {med_e7_4:.3f}')
		st.write(f'Heterossexual: {med_e7_5:.3f}')
		st.write(f'Bissexual: {med_e7_6:.3f}')
		st.write(f'Gay: {med_e7_7:.3f}')
		st.write(f'Lésbica: {med_e7_8:.3f}')
		st.write(f'Gordo(a): {med_e7_9:.3f}')
		st.write(f'Magro(a): {med_e7_10:.3f}')
		st.write(f'Pessoa com deficiência: {med_e7_11:.3f}')
		st.write(f'Branco: {med_e7_12:.3f}')
		st.write(f'Pardo: {med_e7_13:.3f}')
		st.write(f'Preto: {med_e7_14:.3f}')
		st.write(f'Indígena: {med_e7_15:.3f}')
		st.write(f'Rico (a): {med_e7_16:.3f}')
		st.write(f'Pobre: {med_e7_17:.3f}')
		st.write(f'Ateu: {med_e7_18:.3f}')
		st.write(f'Católico(a): {med_e7_19:.3f}')
		st.write(f'Espírita: {med_e7_20:.3f}')
		st.write(f'Evangèlico(a): {med_e7_21:.3f}')
		st.write(f'Religião de matriz africana: {med_e7_22:.3f}')


	st.divider()
	st.subheader('Média da escala dos grupos de pertencimento, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E7_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E7_soma_pertenca'].mean().reset_index()
	media_E7_por_cidade_sexo['Sexo_Cor'] = media_E7_por_cidade_sexo['Sexo'] + " - " + media_E7_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E7_por_cidade_sexo["Cidade"].unique()
	sexos = media_E7_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E7_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E7_por_cidade_sexo[
	    media_E7_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E7_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E7_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E7_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E7_soma_pertenca'].mean().reset_index()
	media_E7_por_cidade_sexo['Sexo_Cor'] = media_E7_por_cidade_sexo['Sexo'] + " - " + media_E7_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E7_por_cidade_sexo["Cidade"].unique()
	sexos = media_E7_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E7_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E7_por_cidade_sexo[
	    media_E7_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E7_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E7_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E7_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E7_soma_pertenca"].mean()

	        fig_media_E7 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E7_soma_pertenca",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E7_soma_pertenca"].round(2),
	            title="Escala 7: Média do número dos grupos de pertencimento, por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E7_soma_pertenca": "Média do número dos grupos de pertencimento",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E7.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 7: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E7.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média do número dos grupos de pertencimento(Escala 7)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E7.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E7, use_container_width=True)

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
	

	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E7 = [
	    'E7_soma_pertenca', 'Escala_7_1', 'Escala_7_2',
	    'Escala_7_3', 'Escala_7_4', 'Escala_7_5',
	    'Escala_7_6', 'Escala_7_7', 'Escala_7_8', 'Escala_7_9',
	    'Escala_7_10', 'Escala_7_11', 'Escala_7_12', 'Escala_7_13', 
	    'Escala_7_14', 'Escala_7_15', 'Escala_7_16', 'Escala_7_17', 
	    'Escala_7_18', 'Escala_7_19', 'Escala_7_20', 'Escala_7_21', 
	    'Escala_7_22'
	]
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc7"):
		# === Interface de seleção ===
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E7, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
	

		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')




# # ----------------------------Escala 8-----------------------------------------------
# # ----------------------------Escala 8-----------------------------------------------
# # ----------------------------Escala 8-----------------------------------------------





elif page == 'Escala 8':
	st.subheader('Interações intergrupais ocorridas no ambiente escolar')

	dados = pd.read_csv('bd_oppes.csv')
	med_e8 = dados['E8_qualid_relacoes'].mean()
	med_e8_1 = dados['Escala_8_1'].mean()
	med_e8_2 = dados['Escala_8_2'].mean()
	med_e8_3 = dados['Escala_8_3'].mean()
	med_e8_4 = dados['Escala_8_4'].mean()
	
	


	st.markdown(f'**Média geral: {med_e8:.3f}**')

	st.divider()
	st.subheader('Média de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 8", key="med_E8"):
		st.write(f'1 = {med_e8_1:.3f} (Alunos(as) de diferentes grupos (raças, etnias, religiões, orientação sexual, gênero, pessoas com deficiência etc.) andam juntos)')
		st.write(f'2 = {med_e8_2:.3f} (Alunos(as) de diferentes grupos (raças, etnias, religiões, orientação sexual, gênero, pessoas com deficiência etc.) confiam uns nos outros)')
		st.write(f'3 = {med_e8_3:.3f} (Alunos(as) de diferentes grupos (raças, etnias religiões, orientação sexual, gênero, pessoas com deficiência etc.) se dão bem)')
		st.write(f'4 = {med_e8_4:.3f} (Alunos(as) de diferentes grupos (raças, etnias, religiões, orientação sexual, gênero, pessoas com deficiência etc.) trabalham juntos em sala de aula)')
		
		

	st.divider()
	st.subheader('Média da escala, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E8_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E8_qualid_relacoes'].mean().reset_index()
	media_E8_por_cidade_sexo['Sexo_Cor'] = media_E8_por_cidade_sexo['Sexo'] + " - " + media_E8_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E8_por_cidade_sexo["Cidade"].unique()
	sexos = media_E8_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E8_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E8_por_cidade_sexo[
	    media_E8_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E8_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E8_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E8_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E8_qualid_relacoes'].mean().reset_index()
	media_E8_por_cidade_sexo['Sexo_Cor'] = media_E8_por_cidade_sexo['Sexo'] + " - " + media_E8_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E8_por_cidade_sexo["Cidade"].unique()
	sexos = media_E8_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E8_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E8_por_cidade_sexo[
	    media_E8_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E8_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E8_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E8_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E8_qualid_relacoes"].mean()

	        fig_media_E8 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E8_qualid_relacoes",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E8_qualid_relacoes"].round(2),
	            title="Escala 1: Média das interações intergrupais, por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E8_qualid_relacoes": "Média das interações intergrupais",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E8.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 8: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E8.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média das interações intergrupais (Escala 8)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E8.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E8, use_container_width=True)

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
	

	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E8 = [
	    'E8_qualid_relacoes', 'Escala_8_1', 'Escala_8_2',
	    'Escala_8_3', 'Escala_8_4'
	]
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc8"):
		# === Interface de seleção ===
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E8, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
	

		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')






# # ----------------------------Escala 9-----------------------------------------------
# # ----------------------------Escala 9-----------------------------------------------
# # ----------------------------Escala 9-----------------------------------------------




elif page == 'Escala 9':
	st.subheader('Situações de interações entre os grupos nas escolas')
	dados = pd.read_csv('bd_oppes.csv')
	#dados = load_dados()

	med_e9 = dados['E9_trat_desig_grupos'].mean()
	med_e9_1 = dados['Escala_9_1'].mean()
	med_e9_2 = dados['Escala_9_2'].mean()
	med_e9_3 = dados['Escala_9_3'].mean()
	med_e9_4 = dados['Escala_9_4'].mean()
	med_e9_5 = dados['Escala_9_5'].mean()
	med_e9_6 = dados['Escala_9_6'].mean()
	med_e9_7 = dados['Escala_9_7'].mean()
	med_e9_8 = dados['Escala_9_8'].mean()
	med_e9_9 = dados['Escala_9_9'].mean()

	st.markdown(f'**Média geral: {med_e9:.3f}**')

	st.divider()
	st.subheader('Média de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 9", key="med_E9"):
		st.write(f'1 = {med_e9_1:.3f} (Os(as) professores(as) e direção/coordenação tratam os(as) alunos(as) de diferentes grupos (raças, etnias, religiões, orientação sexual, gênero, pessoas com deficiência etc.) de forma justa e igualitária)')
		st.write(f'2 = {med_e9_2:.3f} (Os grupos que VOCÊ faz parte (racial, étnico, religioso, de orientação sexual, gênero, pessoas com deficiência etc) são vistos de forma negativa)')
		st.write(f'3 = {med_e9_3:.3f} (Os(as) professores(as) são preconceituosos com diferentes grupos (raças, etnias, religiões, orientação sexual, gênero, pessoas com deficiência etc.))')
		st.write(f'4 = {med_e9_4:.3f} (Seu grupo (racial, étnico, religioso, de orientação sexual, gênero, pessoas com deficiência etc) é representado de forma negativa nos livros didáticos ou outros materiais escolares)')
		st.write(f'5 = {med_e9_5:.3f} (Seus(suas) colegas gostam de ter amigos de diferentes raças, etnias, religiões, orientação sexual e de gênero etc.)')
		st.write(f'6 = {med_e9_6:.3f} (Os(as) professores(as) e direção/coordenação valorizam pessoas de diferentes raças, etnias, religiões, orientação sexual, gênero, pessoas com deficiência etc.)')
		st.write(f'7 = {med_e9_7:.3f} (Os(as) professores(as) e direção/coordenação incentivam que os(as) alunos(as) tenham amigos(as) de diferentes raças, etnias, religiões, orientação sexual, de gênero, pessoas com deficiência etc.)')
		st.write(f'8 = {med_e9_8:.3f} (Os(as) alunos(as) acham que é bom estudar com pessoas de diferentes raças, etnias, religiões, orientação sexual, de gênero, pessoas com deficiência etc.)')
		st.write(f'9 = {med_e9_9:.3f} (Você tem oportunidades de aprender sobre a cultura de diferentes grupos (pessoas de outras raças, etnias, religião, orientação sexual, de gênero, pessoas com deficiência etc.)')

	st.divider()
	st.subheader('Média da escala, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E9_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E9_trat_desig_grupos'].mean().reset_index()
	media_E9_por_cidade_sexo['Sexo_Cor'] = media_E9_por_cidade_sexo['Sexo'] + " - " + media_E9_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E9_por_cidade_sexo["Cidade"].unique()
	sexos = media_E9_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E9_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E9_por_cidade_sexo[
	    media_E9_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E9_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E9_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E9_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E9_trat_desig_grupos'].mean().reset_index()
	media_E9_por_cidade_sexo['Sexo_Cor'] = media_E9_por_cidade_sexo['Sexo'] + " - " + media_E9_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E9_por_cidade_sexo["Cidade"].unique()
	sexos = media_E9_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E9_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E9_por_cidade_sexo[
	    media_E9_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E9_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E9_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E9_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E9_trat_desig_grupos"].mean()

	        fig_media_E9 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E9_trat_desig_grupos",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E9_trat_desig_grupos"].round(2),
	            title="Escala 9: Média de interações entre os grupos, por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E9_trat_desig_grupos": "Média das interações entre os grupos",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E9.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 9: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E9.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média  (Escala 1)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E9.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E9, use_container_width=True)

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
	

	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E9 = [
	    'E9_trat_desig_grupos', 'Escala_9_1', 'Escala_9_2',
	    'Escala_9_3', 'Escala_9_4', 'Escala_9_5',
	    'Escala_9_6', 'Escala_9_7', 
	    'Escala_9_8', 'Escala_9_9'
	]
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc9"):
		# === Interface de seleção ===
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E9, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
	

		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')




# # ----------------------------Escala 10-----------------------------------------------
# # ----------------------------Escala 10-----------------------------------------------
# # ----------------------------Escala 10-----------------------------------------------



elif page == 'Escala 10':
	st.subheader('Estados emocionais negativos relatados')

	dados = pd.read_csv('bd_oppes.csv')
	med_e10 = dados['E10_est_emoc_neg'].mean()
	med_e10_1 = dados['Escala_10_1'].mean()
	med_e10_2 = dados['Escala_10_2'].mean()
	med_e10_3 = dados['Escala_10_3'].mean()
	med_e10_4 = dados['Escala_10_4'].mean()
	med_e10_5 = dados['Escala_10_5'].mean()
	med_e10_6 = dados['Escala_10_6'].mean()
	
	st.markdown(f'**Média geral: {med_e10:.3f}**')

	st.divider()
	st.subheader('Média de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 10", key="med_e10"):
		st.write(f'1 = {med_e10_1:.3f} (Pensamentos de acabar com a vida.)')
		st.write(f'2 = {med_e10_2:.3f} (Sentir-se sozinha(o))')
		st.write(f'3 = {med_e10_3:.3f} (Sentir-se triste)')
		st.write(f'4 = {med_e10_4:.3f} (Não ter interesse por nada)')
		st.write(f'5 = {med_e10_5:.3f} (Sentir-se sem esperança perante o futuro)')
		st.write(f'6 = {med_e10_6:.3f} (Sentir que não tem valor)')
		

	st.divider()
	st.subheader('Média da escala dos estados emocionais negativos, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E10_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E10_est_emoc_neg'].mean().reset_index()
	media_E10_por_cidade_sexo['Sexo_Cor'] = media_E10_por_cidade_sexo['Sexo'] + " - " + media_E10_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E10_por_cidade_sexo["Cidade"].unique()
	sexos = media_E10_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E10_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E10_por_cidade_sexo[
	    media_E10_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E10_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E10_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E10_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E10_est_emoc_neg'].mean().reset_index()
	media_E10_por_cidade_sexo['Sexo_Cor'] = media_E10_por_cidade_sexo['Sexo'] + " - " + media_E10_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E10_por_cidade_sexo["Cidade"].unique()
	sexos = media_E10_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E10_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E10_por_cidade_sexo[
	    media_E10_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E10_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E10_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E10_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E10_est_emoc_neg"].mean()

	        fig_media_E10 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E10_est_emoc_neg",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E10_est_emoc_neg"].round(2),
	            title="Escala 10: Média dos estados emocionais negativos, por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E10_est_emoc_neg": "Média dos estados emocionais negativos",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E10.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 10: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E10.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média dos estados emocionais negativos (Escala 10)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E10.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E10, use_container_width=True)

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
	

	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E10 = [
	    'E10_est_emoc_neg', 'Escala_10_1', 'Escala_10_2',
	    'Escala_10_3', 'Escala_10_4', 'Escala_10_5',
	    'Escala_10_6'
	]
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc10"):
		# === Interface de seleção ===
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E10, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
	

		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')



# # ----------------------------Escala 11-----------------------------------------------
# # ----------------------------Escala 11-----------------------------------------------
# # ----------------------------Escala 11-----------------------------------------------


elif page == 'Escala 11':
	st.subheader('Relatos dos estudantes sobre a satisfação com a vida')

	dados = pd.read_csv('bd_oppes.csv')
	med_e11 = dados['E11_satisf_vida'].mean()
	med_e11_1 = dados['Escala_11_1'].mean()
	med_e11_2 = dados['Escala_11_2'].mean()
	med_e11_3 = dados['Escala_11_3'].mean()
	med_e11_4 = dados['Escala_11_4'].mean()
	med_e11_5 = dados['Escala_11_5'].mean()

	
	st.markdown(f'**Média geral: {med_e11:.3f}**')

	st.divider()
	st.subheader('Média de cada item da escala')


	if st.checkbox ("Marque aqui para visualizar a média de cada item da Escala 10", key="med_e11"):
		st.write(f'1 = {med_e11_1:.3f} (A minha vida está próxima do meu ideal)')
		st.write(f'2 = {med_e11_2:.3f} (As minhas condições de vida são excelentes)')
		st.write(f'3 = {med_e11_3:.3f} (Eu estou satisfeita (o) com a minha vida)')
		st.write(f'4 = {med_e11_4:.3f} (Até agora eu tenho conseguido as coisas importantes que eu quero na vida)')
		st.write(f'5 = {med_e11_5:.3f} (Se eu pudesse viver a minha vida de novo eu não mudaria quase nada)')
		

	st.divider()
	st.subheader('Média de satisfação com a vida, por cidade, sexo e cor da pele')   


# Calcular média por cidade, sexo e cor da pele

# Calcular média
	media_E11_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E11_satisf_vida'].mean().reset_index()
	media_E11_por_cidade_sexo['Sexo_Cor'] = media_E11_por_cidade_sexo['Sexo'] + " - " + media_E11_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E11_por_cidade_sexo["Cidade"].unique()
	sexos = media_E11_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E11_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros baseados no estado
	df_filtrado = media_E11_por_cidade_sexo[
	    media_E11_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E11_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E11_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()
	   

	# 🧮 Agrupamento e preparação dos dados
	media_E11_por_cidade_sexo = dados.groupby(['Cidade', 'Sexo', 'Cor_da_pele'])['E11_satisf_vida'].mean().reset_index()
	media_E11_por_cidade_sexo['Sexo_Cor'] = media_E11_por_cidade_sexo['Sexo'] + " - " + media_E11_por_cidade_sexo['Cor_da_pele']

	# Valores únicos
	cidades = media_E11_por_cidade_sexo["Cidade"].unique()
	sexos = media_E11_por_cidade_sexo["Sexo"].unique()
	cores_pele = media_E11_por_cidade_sexo["Cor_da_pele"].unique()

	# Inicializar session_state
	if "cidade_selecionada" not in st.session_state:
	    st.session_state["cidade_selecionada"] = list(cidades)
	if "sexo_selecionado" not in st.session_state:
	    st.session_state["sexo_selecionado"] = list(sexos)
	if "cor_selecionada" not in st.session_state:
	    st.session_state["cor_selecionada"] = list(cores_pele)

	# Aplicar filtros
	df_filtrado = media_E11_por_cidade_sexo[
	    media_E11_por_cidade_sexo["Cidade"].isin(st.session_state["cidade_selecionada"]) &
	    media_E11_por_cidade_sexo["Sexo"].isin(st.session_state["sexo_selecionado"]) &
	    media_E11_por_cidade_sexo["Cor_da_pele"].isin(st.session_state["cor_selecionada"])
	].copy()

	# Mostrar gráfico (opcional)
	if st.checkbox("Marque aqui para visualizar o gráfico das médias, por cidade, sexo e cor da pele", key="graf_E11_csc"):
	    if df_filtrado.empty:
	        st.warning("Nenhum dado disponível para os filtros selecionados.")
	    else:
	        df_filtrado["Sexo_Cor"] = df_filtrado["Sexo"] + " - " + df_filtrado["Cor_da_pele"]
	        media_geral = df_filtrado["E11_satisf_vida"].mean()

	        fig_media_E11 = px.bar(
	            df_filtrado,
	            x="Cidade",
	            y="E11_satisf_vida",
	            color="Sexo_Cor",
	            barmode="group",
	            text=df_filtrado["E11_satisf_vida"].round(2),
	            title="Escala 10: Média de satisfação com a vida, por Cidade, Sexo e Cor da Pele",
	            labels={
	                "Cidade": "Cidade",
	                "E11_satisf_vida": "Média de satisfação com a vida",
	                "Sexo_Cor": "Sexo e Cor da Pele"
	            }
	        )

	        fig_media_E11.add_hline(
	            y=media_geral,
	            line_dash="dot",
	            line_color="red",
	            annotation_text=f"Média Geral da Escala 11: {media_geral:.3f}",
	            annotation_position="top left"
	        )

	        fig_media_E11.update_layout(
	            xaxis_title="Cidade",
	            yaxis_title="Média de satisfação com a vida (Escala 11)",
	            legend_title="Sexo e Cor da Pele",
	            plot_bgcolor="#F9F9F9",
	            bargap=0.15,
	        )

	        fig_media_E11.update_traces(
	            textposition="outside",
	            marker_line_width=0.5
	        )

	        st.plotly_chart(fig_media_E11, use_container_width=True)

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
	

	# === Carregamento e pré-processamento dos dados ===
	dados = pd.read_csv('bd_oppes.csv')
	dados = dados.drop(['Cidade'], axis=1)

	# Colunas numéricas de interesse (escalas)
	colunas_escalas_E11 = [
	    'E11_satisf_vida', 'Escala_11_1', 'Escala_11_2',
	    'Escala_11_3', 'Escala_11_4', 'Escala_11_5'
	]
	if st.checkbox ("Marque aqui para selecionar as variáveis e visualizar os resultados ", key="graf_reg_esc11"):
		# === Interface de seleção ===
		st.write('Use o filtro abaixo para selecionar as variáveis')
		x_axis = 'Idade'
		y_axis = st.selectbox('Selecione a variável a ser incluída no eixo Y', colunas_escalas_E11, index=0)

		# Conversão e limpeza dos dados
		dados[x_axis] = pd.to_numeric(dados[x_axis], errors='coerce')
		dados[y_axis] = pd.to_numeric(dados[y_axis], errors='coerce')
		dados = dados.dropna(subset=[x_axis, y_axis])
		dados = dados[np.isfinite(dados[x_axis]) & np.isfinite(dados[y_axis])]

		# Verificação de dados suficientes
		if dados.empty:
		    st.warning("Não há dados suficientes para realizar a regressão linear.")
		    st.stop()

		# === Cálculo da linha de regressão ===
		m, b = np.polyfit(dados[x_axis], dados[y_axis], 1)
		x_regressao = np.array(dados[x_axis])
		y_regressao = m * x_regressao + b

		# === Criação do gráfico ===
		fig = px.scatter(
		    dados, x=x_axis, y=y_axis,
		    color='Cor_da_pele', symbol='Sexo',
		    title='Gráfico de Dispersão com Linha de Regressão.'
		)
		fig.add_trace(go.Scatter(
		    x=x_regressao, y=y_regressao,
		    mode='lines', name='Linha de Regressão',
		    line=dict(color='red', width=3)
		))
	

		st.plotly_chart(fig)

		# === Regressão linear com scikit-learn ===
		X = dados[[x_axis]].values
		y = dados[y_axis].values
		model = LinearRegression()
		model.fit(X, y)

		# Resultados do modelo
		coeficiente = model.coef_[0]
		intercepto = model.intercept_
		r2 = model.score(X, y)

		# === Exibição dos resultados ===
		st.write(f'**Coeficiente de regressão (inclinação):** {coeficiente:.3f}')
		st.write(f'**Intercepto:** {intercepto:.3f}')
		st.write(f'**Coeficiente de determinação (R²):** {r2:.3f}')


#--------------modelos de regressão
#--------------modelos de regressão
#--------------modelos de regressão
elif page == 'Modelos de regressão':

# Título
	st.subheader('Regressão Linear Múltipla com Padronização, Outliers e Resultados Originais')
	st.write("Selecione variáveis para regressão, com remoção de outliers e visualização dos resultados em escalas padronizada e original.")

	# Carregamento dos dados
	dados = pd.read_csv('bd_oppes.csv')

	# Seleção apenas das colunas desejadas
	colunas_desejadas = ['E1_ameacas', 'E2_situacoes_estresse', 'E3_4_agentes', 'E5_disc_pessoal', 'E7_soma_pertenca', 'E7_soma_pertenca', 'E8_qualid_relacoes', 'E9_trat_desig_grupos', 'E10_est_emoc_neg', 'E11_satisf_vida']
	colunas_disponiveis = [col for col in colunas_desejadas if col in dados.columns]

	# Interface de seleção
	variavel_dependente = st.selectbox("Selecione a variável dependente (Y):", colunas_disponiveis)
	colunas_independentes = st.multiselect(
	    "Selecione uma ou mais variáveis independentes (X):",
	    [col for col in colunas_disponiveis if col != variavel_dependente],
	    placeholder="Selecione uma ou mais opções"
	)

	# Descrições das variáveis
	descricoes = {
	    'E1_ameacas': 'E1_ameacas: Frequência de exposição a ameaças no ambiente escolar.',
	    'E2_situacoes_estresse': 'E2_situacoes_estresse: Número de situações estressantes enfrentadas.',
	    'E3_4_agentes': 'E3_4_agentes: Ação dos professores e da escola.',
	    'E5_disc_pessoal': 'E5_disc_pessoal: Qualidade da disciplina pessoal observada.',
	    'E6_locais': 'E6_locais: Avaliação da segurança e conforto dos locais da escola.',
	    'E7_soma_pertenca': 'E7_soma_pertenca: Soma dos grupos de pertencimento.',
	    'E8_qualid_relacoes': 'E8_qualid_relacoes: Qualidade das relações entre os grupos.',
	    'E9_trat_desig_grupos': 'E9_trat_desig_grupos: Tratamento desigual entre os grupos.',
	    'E10_est_emoc_neg': 'E10_est_emoc_neg: Estados emocionais negativos.',
	    'E11_satisf_vida': 'E11_satisf_vida: Satisfação com a vida.'

	}

	# Mostrar descrição da variável dependente
	st.markdown(f"**Descrição da variável dependente:** {descricoes.get(variavel_dependente, 'Sem descrição disponível.')}")

	# Mostrar descrições das independentes
	if colunas_independentes:
	    st.markdown("**Descrição das variáveis independentes:**")
	    for var in colunas_independentes:
	        st.markdown(f"- {descricoes.get(var, 'Sem descrição disponível.')}")
	else:
	    st.warning("Selecione pelo menos uma variável independente.")
	    st.stop()

	# Remoção de NA
	X = dados[colunas_independentes].apply(pd.to_numeric, errors='coerce')
	y = dados[variavel_dependente].apply(pd.to_numeric, errors='coerce')
	dados_validos = pd.concat([X, y], axis=1).dropna()

	if dados_validos.empty:
	    st.warning("Dados insuficientes após remoção de NAs.")
	    st.stop()

	# Padronização
	scaler_X = StandardScaler()
	scaler_y = StandardScaler()

	X_pad = scaler_X.fit_transform(dados_validos[colunas_independentes])
	y_pad = scaler_y.fit_transform(dados_validos[[variavel_dependente]])

	# Remoção de outliers
	z_scores = np.abs(np.concatenate([X_pad, y_pad], axis=1))
	filtros = (z_scores < 3).all(axis=1)
	X_filtrado = X_pad[filtros]
	y_filtrado = y_pad[filtros].flatten()

	# Salva os dados originais filtrados
	dados_filtrados_originais = dados_validos.iloc[filtros]

	if len(X_filtrado) < len(colunas_independentes) + 1:
	    st.warning("Poucos dados após remoção de outliers.")
	    st.stop()


	st.divider()
	# Regressão
	modelo = LinearRegression()
	modelo.fit(X_filtrado, y_filtrado)
	y_prev_pad = modelo.predict(X_filtrado)
	residuos_pad = y_filtrado - y_prev_pad
	r2_pad = modelo.score(X_filtrado, y_filtrado)

	# Inversão dos valores previstos para escala original
	y_prev_orig = scaler_y.inverse_transform(y_prev_pad.reshape(-1, 1)).flatten()
	y_real_orig = dados_filtrados_originais[variavel_dependente].values
	residuos_orig = y_real_orig - y_prev_orig

	# === Resultados Padronizados ===
	st.markdown("### Resultados Padronizados")
	st.write(f"**Intercepto (padronizado):** {modelo.intercept_:.3f}")
	st.write(f"**R² (padronizado):** {r2_pad:.3f}")

	st.markdown("**Coeficientes padronizados:**")
	for var, coef in zip(colunas_independentes, modelo.coef_):
	    st.write(f"- {var}: {coef:.3f}")

	# === Resultados Originais ===
	# st.markdown("### Resultados na Escala Original")
	# st.write("Esses valores são obtidos ao reverter a padronização das previsões.")
	# df_resultados = dados_filtrados_originais.copy()
	# df_resultados['Previsto (original)'] = y_prev_orig
	# df_resultados['Resíduo (original)'] = residuos_orig
	# st.dataframe(df_resultados[[variavel_dependente, 'Previsto (original)', 'Resíduo (original)']].round(3))

	# === Gráfico de Resíduos Originais ===
	st.markdown("### Gráfico de Resíduos (Original)")
	fig_residuos = px.scatter(
	    x=y_prev_orig, y=residuos_orig,
	    labels={'x': 'Previsto (Ŷ)', 'y': 'Resíduo'},
	    title='Gráfico de Resíduos na Escala Original'
	)
	fig_residuos.add_hline(y=0, line_dash="dash", line_color="red")
	st.plotly_chart(fig_residuos)

	st.divider()


	# ==========================
	# 1. REGRESSÃO com STATSMODELS
	# ==========================
	X_orig = dados_filtrados_originais[colunas_independentes]
	y_orig = dados_filtrados_originais[variavel_dependente]

	# Adiciona constante para intercepto
	X_sm = sm.add_constant(X_orig)
	modelo_sm = sm.OLS(y_orig, X_sm).fit()

	st.markdown("## 📊 Análise Estatística (Statsmodels)")
	st.text(modelo_sm.summary())

	# ==========================
	# 2. MULTICOLINEARIDADE (VIF)
	# ==========================
	st.markdown("### 🔄 Multicolinearidade (VIF)")

	# Calcular VIF para cada variável
	vif_data = pd.DataFrame()
	vif_data["Variável"] = X_sm.columns
	vif_data["VIF"] = [variance_inflation_factor(X_sm.values, i) for i in range(X_sm.shape[1])]
	st.dataframe(vif_data.round(2))

	# ==========================
	# 3. NORMALIDADE DOS RESÍDUOS
	# ==========================
	st.markdown("### 📈 Teste de Normalidade dos Resíduos (Shapiro-Wilk)")

	residuos_statsmodels = modelo_sm.resid
	stat, p_val_shapiro = shapiro(residuos_statsmodels)
	st.write(f"**Estatística de Shapiro-Wilk:** {stat:.4f}")
	st.write(f"**p-valor:** {p_val_shapiro:.4f}")
	if p_val_shapiro > 0.05:
	    st.success("Não há evidências de que os resíduos violem a normalidade (p > 0.05).")
	else:
	    st.error("Os resíduos podem não ser normalmente distribuídos (p < 0.05).")

	# ==========================
	# 4. HETEROCEDASTICIDADE (Breusch-Pagan)
	# ==========================
	st.markdown("### 📉 Teste de Heterocedasticidade (Breusch-Pagan)")

	bp_test = het_breuschpagan(residuos_statsmodels, X_sm)
	bp_labels = ['Estatística LM', 'p-valor LM', 'Estatística F', 'p-valor F']
	for label, val in zip(bp_labels, bp_test):
	    st.write(f"{label}: {val:.4f}")
	if bp_test[1] > 0.05:
	    st.success("Não há evidência de heterocedasticidade (p > 0.05).")
	else:
	    st.error("Pode haver heterocedasticidade (p < 0.05).")


else: 
	st.subheader('Dados Textuais')

	cidades = {
	    "de todos os participantes": "rede_semantica_palavras.png",
	    "dos participantes de Aracaju": "rede_aracaju.png",
	    "dos participantes de Campos de Brito": "rede_campos_brito.png",
	    "dos participantes de Canindé do São Francisco": "rede_caninde.png",
	    "dos participantes de Nossa Senhora do Carmo": "rede_carmo.png",
	    "dos participantes de Nossa Senhora das Dores": "rede_dores.png",
	    "dos participantes de Estância": "rede_estancia.png",
	    "dos participantes de Neópolis": "rede_neopolis.png",
	    "dos participantes de Porto da Folha": "rede_porto_folha.png",
	    "dos participantes de Simão Dias": "rede_simao_dias.png"
	}

	for cidade, imagem in cidades.items():
	    with st.expander(f"Visualizar as respostas {cidade}"):
	        st.image(imagem, caption=f"Rede semântica das respostas {cidade} sobre assédio na Internet")
