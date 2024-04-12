import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader
from scipy.stats import gaussian_kde

def medianOfNumericColsByASpecificCol(numeric_cols, column, fileName, config=False):
    plt.figure()  # Criar uma nova figura
    # Calcular a mediana apenas nas colunas numéricas
    median = numeric_cols.groupby(df_without_hash[column]).median()

    # Plotar o gráfico de barras
    ax = median.plot.bar()

    # Adicionar rótulos com os valores das barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    #plt.ylabel(title)
    if(config):
        plt.title(config['title'])

    plt.xticks(rotation=0)
    plt.savefig(fileName+'.png')  

def meanOfNumericColsByASpecificCol(numeric_cols, column, fileName, config=False):
    plt.figure()  # Criar uma nova figura
    # Calcular a média apenas nas colunas numéricas
    mean = numeric_cols.groupby(df_without_hash[column]).mean()

    # Plotar o gráfico de barras
    ax = mean.plot.bar()

    # Adicionar rótulos com os valores das barras
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', xytext=(0, 5), 
                    textcoords='offset points')

    #plt.ylabel(title)
    if(config):
        plt.title(config['title'])

    plt.xticks(rotation=0)
    plt.savefig(fileName+'.png')

def findIncomeOutlier(dataFrame):
    plt.figure()  # Criar uma nova figura
    plt.figure(figsize=(8,6))
    plt.boxplot(df['Renda anual'])

    # Adicionar título e rótulo ao eixo x
    plt.title('Boxplot da Renda Anual')
    plt.xlabel('Renda Anual')

    plt.savefig('box_plot_renda.png') 

def relacaoEntreVariaveis(variavel1, variavel2, title, fileName):
    plt.figure()  # Criar uma nova figura
    # Calcular a linha de tendência usando regressão linear simples
    coeficientes = np.polyfit(df[variavel1], df[variavel2], 1)
    linha_tendencia = np.poly1d(coeficientes)

    # Plotar o gráfico de dispersão de idade vs renda anual
    plt.figure()  # Criar uma nova figura
    plt.scatter(df[variavel1], df[variavel2])
    plt.xlabel(variavel1)
    plt.ylabel(variavel2)
    plt.title(title)

    # Plotar a linha de tendência
    plt.plot(df[variavel1], linha_tendencia(df[variavel1]), color='red')

    plt.savefig(fileName+'.png')  

def graficoProporcaoGenero(proporcao_genero):
    plt.figure()  # Criar uma nova figura
    fig, ax = plt.subplots()
    labels = proporcao_genero.index
    sizes = proporcao_genero.values
    colors = ['lightblue', 'lightcoral']

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)

    ax.set_title('Proporção por Gênero')

    # Exibir o gráfico
    plt.savefig('grafico_proporcao_mulheres_homens_pie.png')

def proporcao_grad_posgrad_por_genero(df):
    # Filtrando o DataFrame para incluir apenas 'Graduação' e 'Pós-Graduação'
    df_filtered = df[df['Nível de educação'].isin(['Graduação', 'Pós-graduação'])]

    # Agrupando por 'Gênero' e contando o número de ocorrências de 'Graduação' e 'Pós-Graduação'
    grouped = df_filtered.groupby('Gênero')['Nível de educação'].value_counts().unstack(fill_value=0)

    # Obtendo o número total de homens e mulheres no DataFrame original
    total_homens = df[df['Gênero'] == 'Masculino'].shape[0]
    total_mulheres = df[df['Gênero'] == 'Feminino'].shape[0]

    # Calculando a proporção de homens e mulheres com graduação e pós-graduação
    proporcao_homens_grad = grouped.loc['Masculino', 'Graduação'] / total_homens
    proporcao_homens_posgrad = grouped.loc['Masculino', 'Pós-graduação'] / total_homens

    proporcao_mulheres_grad = grouped.loc['Feminino', 'Graduação'] / total_mulheres
    proporcao_mulheres_posgrad = grouped.loc['Feminino', 'Pós-graduação'] / total_mulheres

    return {
        'homens_graduacao': proporcao_homens_grad,
        'homens_pos_graduacao': proporcao_homens_posgrad,
        'mulheres_graduacao': proporcao_mulheres_grad,
        'mulheres_pos_graduacao': proporcao_mulheres_posgrad
    }

def verificacao_renda_anual_profissao(df):
    # Agrupando o DataFrame pela profissão e calculando a média da renda anual para cada grupo
    df_grouped = df.groupby('Profissão')['Renda anual'].mean().sort_values(ascending=False)


    # Plotando o gráfico de barras horizontal
    plt.figure(figsize=(10, 6))
    plt.barh(df_grouped.index, df_grouped.values)
    plt.xlabel('Renda Anual')
    plt.ylabel('Profissão')
    plt.title('Renda Anual por Profissão (Ordenado)')
    plt.tight_layout()
    plt.savefig('grafico_profissao_renda.png')


def profissao_por_genero(df, gender):
    # Filtrando o DataFrame pelo gênero especificado
    df_gender = df[df['Gênero'] == gender]

    # Contando o número de ocorrências de cada profissão
    profession_count = df_gender['Profissão'].value_counts()

    profession_count_sorted = profession_count.sort_index()

    # Plotando o gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(profession_count_sorted.index, profession_count_sorted.values)
    plt.xlabel('Profissão')
    plt.ylabel(f'Quantidade de {gender}s')
    plt.title(f'Quantidade de {gender}s por Profissão')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{gender}_profissao.png')

# Carregar o dataset do Excel
df = pd.read_excel('Sistematização.xlsx')
df = df.drop_duplicates(subset=[col for col in df.columns if col != '#'])
# Ajustar a proporção da renda anual dividindo por mil
df_without_hash = df.drop(columns=['#'])
df_without_hash['Renda anual'] /= 10000
df_without_hash['Idade'] /= 10
df_without_hash['Horas de trabalho por semana'] /= 10
df_without_hash['Nível de satisfação com o trabalho atual'] /= 10

# Selecionar apenas as colunas numéricas
numeric_cols = df_without_hash.select_dtypes(include='number')

############################################
### IDENTIFICANDO NECESSIDADES DE AJUSTE ###
############################################
findIncomeOutlier(df)
#df_info = df.info()
#buffer = io.StringIO()
#df.info(buf=buffer)
#df_info = buffer.getvalue()
df_info = (
    f"<table><caption>RangeIndex: {df.shape[0]} entries, 0 to {df.shape[0] - 1}<br/>"
    f"Data columns (total {df.shape[1]} columns)</caption>"
)

# Adicionando informações de cada coluna
for i, (col, non_null_count, dtype) in enumerate(zip(df.columns, df.notnull().sum(), df.dtypes), 1):
    df_info += f"<tr><td>{i}</td><td>{col:<40}</td><td>{non_null_count}</td><td>non-null</td><td>{dtype}</td></tr>"

# Calculando o uso de memória
memory_usage = df.memory_usage(deep=True).sum() / 1024
df_info += f"""</table>"""
##########################
### DADOS SOBRE GÊNERO ###
##########################
# Calculando proporção de homens e mulheres na amostra
proporcao_genero = df['Gênero'].value_counts(normalize=True)
# Calcular o total absoluto de homens e mulheres
total_absoluto_homens = int(proporcao_genero['Masculino'] * len(df))
total_absoluto_mulheres = int(proporcao_genero['Feminino'] * len(df))

# Definir a porcentagem de homens e mulheres como 50%
total_percentual_homens = proporcao_genero['Masculino'] * 100
total_percentual_mulheres =  proporcao_genero['Feminino'] * 100

graficoProporcaoGenero(proporcao_genero)

#################################
### DADOS ESTATÍSTICOS GERAIS ###
#################################
desc_stats = df.describe()
valores_renda_anual = desc_stats.loc['mean':'max', 'Renda anual']

desc_stats_invertida = desc_stats.T

######################################
### CORRELAÇÃO IDADE x RENDA ANUAL ###
######################################
# Calculando a correlação entre idade e renda anual
correlacao_idade_renda = df['Idade'].corr(df['Renda anual'])

# Calculando a correlação entre renda anual e nível de satisfação com o trabalho atual
correlacao_renda_idade = df['Renda anual'].corr(df['Idade'])

relacaoEntreVariaveis('Idade', 'Renda anual', 'Idade vs Renda Anual','correlacao_renda_idade')

###########################################
### CORRELAÇÃO SATISFAÇÃO x RENDA ANUAL ###
###########################################
# Calculando a correlação entre idade e renda anual
correlacao_satisfacao_renda = df['Nível de satisfação com o trabalho atual'].corr(df['Renda anual'])

# Calculando a correlação entre renda anual e nível de satisfação com o trabalho atual
correlacao_renda_satisfacao = df['Renda anual'].corr(df['Nível de satisfação com o trabalho atual'])

relacaoEntreVariaveis('Nível de satisfação com o trabalho atual', 'Renda anual', 'Nível de satisfação vs Renda Anual','correlacao_renda_satisfacao')


####################################################
### CORRELAÇÃO IDADE x RENDA ANUAL - SEM OUTLIER ###
####################################################
df = df[df['Renda anual'] <= 135000]

# Calculando a correlação entre idade e renda anual
correlacao_idade_renda_sem_outlier = df['Idade'].corr(df['Renda anual'])

# Calculando a correlação entre renda anual e nível de satisfação com o trabalho atual
correlacao_renda_idade_sem_outlier = df['Renda anual'].corr(df['Idade'])

relacaoEntreVariaveis('Idade', 'Renda anual', 'Idade vs Renda Anual (s/ outlier)','correlacao_renda_idade_s_outlier')

#########################################################
### CORRELAÇÃO SATISFAÇÃO x RENDA ANUAL - SEM OUTLIER ###
#########################################################
# Calculando a correlação entre idade e renda anual
correlacao_satisfacao_renda_sem_outlier = df['Nível de satisfação com o trabalho atual'].corr(df['Renda anual'])

# Calculando a correlação entre renda anual e nível de satisfação com o trabalho atual
correlacao_renda_satisfacao_sem_outlier = df['Renda anual'].corr(df['Nível de satisfação com o trabalho atual'])

relacaoEntreVariaveis('Nível de satisfação com o trabalho atual', 'Renda anual', 'Nível de satisfação vs Renda Anual (s/ outlier)','correlacao_renda_satisfacao_s_outlier')


################
### GRÁFICOS ###
################
config = {
    'title': 'Mediana de variáveis relacionadas à Gêneros'
}
medianOfNumericColsByASpecificCol(numeric_cols,'Gênero','analise_barras_genero', config)

config = {
    'title': 'Mediana de variáveis relacionadas à Formação'
}
medianOfNumericColsByASpecificCol(numeric_cols,'Nível de educação', 'analise_barras_formacao',config)

################
### GRÁFICOS ###
################
config = {
    'title': 'Média de variáveis relacionadas à Gêneros'
}
meanOfNumericColsByASpecificCol(numeric_cols,'Gênero','analise_barras_genero_media', config)

config = {
    'title': 'Média de variáveis relacionadas à Formação'
}
meanOfNumericColsByASpecificCol(numeric_cols,'Nível de educação', 'analise_barras_formacao_media',config)

verificacao_renda_anual_profissao(df)
profissao_por_genero(df, 'Feminino')
profissao_por_genero(df, 'Masculino')
#plt.hist(df['Renda anual'], bins=10, density=True, alpha=0.7, color='blue')
#plt.xlabel('Renda anual')
#plt.ylabel('Densidade')

def histograma():
    plt.figure()
    histogramaRendaAnual = sns.histplot(data = df, x = "Renda anual", kde = True)
    histogramaRendaAnual.figure.savefig('histograma_renda_anual.png')

histograma()

proporcao_genero_formacao = proporcao_grad_posgrad_por_genero(df)
####################
### CRIANDO HTML ###
####################
# Manipulando Arquivo HTML do Relatório Final
nome_arquivo_html = 'relatorio_analise'
env = Environment(loader=FileSystemLoader('./'))

# Carregue o arquivo HTML como um template
template = env.get_template(nome_arquivo_html+'.html')


# Preparar os dados estatísticos
dados_estatisticos = desc_stats_invertida.to_html()

# Definir os dados
dados = {
    'info_variaveis': df_info,
    'total_absoluto_homens': total_absoluto_homens,
    'total_percentual_homens': total_percentual_homens,
    'total_absoluto_mulheres': total_absoluto_mulheres,
    'total_percentual_mulheres': total_percentual_mulheres,
    'dados_estatisticos': dados_estatisticos,
    'correlacao_idade_renda': correlacao_idade_renda,
    'correlacao_renda_idade': correlacao_renda_idade,
    'correlacao_satisfacao_renda': correlacao_satisfacao_renda,
    'correlacao_renda_satisfacao': correlacao_renda_satisfacao,
    'correlacao_idade_renda_sem_outlier': correlacao_idade_renda_sem_outlier,
    'correlacao_renda_idade_sem_outlier': correlacao_renda_idade_sem_outlier,
    'correlacao_satisfacao_renda_sem_outlier': correlacao_satisfacao_renda_sem_outlier,
    'correlacao_renda_satisfacao_sem_outlier': correlacao_renda_satisfacao_sem_outlier,
    'renda_minima': "{:,.2f}".format(valores_renda_anual['min']),
    'renda_maxima': "{:,.2f}".format(valores_renda_anual['max']),
    'desvio_padrao_renda': valores_renda_anual['std'],
    'homens_graduacao': round(proporcao_genero_formacao['homens_graduacao'],2)*100,
    'homens_pos_graduacao': round(proporcao_genero_formacao['homens_pos_graduacao'],2)*100,
    'mulheres_graduacao': round(proporcao_genero_formacao['mulheres_graduacao'],2)*100,
    'mulheres_pos_graduacao': round(proporcao_genero_formacao['mulheres_pos_graduacao'],2)*100,
}


# Renderize o template com os dados fornecidos
output = template.render(dados)

# Salve o resultado em um arquivo HTML
with open(nome_arquivo_html+'_final.html', 'w') as f:
    f.write(output)

print("Arquivo HTML gerado com sucesso:", nome_arquivo_html)

