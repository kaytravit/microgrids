import pandas as pd

df = pd.read_csv('SolarPrediction.csv')

#Solar radiation: watts per meter^2
#Temperature: degrees Fahrenheit
#Humidity: percent
#Barometric pressure: Hg
#Wind direction: degrees
#Wind speed: miles per hour
#Sunrise/sunset: Hawaii time


#tratamento de dados
# Converta a coluna de data para o tipo datetime
df['Data'] = pd.to_datetime(df['Data'], format='%m/%d/%Y %I:%M:%S %p')

# Extraia o dia, mês e ano em colunas separadas
df['dia'] = df['Data'].dt.day
df['mes'] = df['Data'].dt.month
df['ano'] = df['Data'].dt.year

# Crie uma nova coluna de data no formato dd/mm/yyyy
df['data_formatada'] = df.apply(lambda row: f"{row['dia']:02d}/{row['mes']:02d}/{row['ano']}", axis=1)

# adicione a coluna 'data_formatada' no lugar da Data
df['Data'] = df['data_formatada']

#excluir coluna data_formatada, dia, mes e ano
df.drop('dia', axis=1, inplace=True)
df.drop('mes', axis=1, inplace=True)
df.drop('ano', axis=1, inplace=True)
df.drop('data_formatada', axis=1, inplace=True)

# Supondo que você já tenha um DataFrame chamado df com a coluna "Data" no formato dd/mm/yyyy
# Crie a coluna de data no formato datetime
df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y')

# Extraia o dia, mês e ano em colunas separadas
df['Dia'] = df['Data'].dt.day
df['Mês'] = df['Data'].dt.month
df['Ano'] = df['Data'].dt.year

# Crie uma nova coluna chamada "Estação do Ano" com as estações do ano correspondentes
# Baseado na divisão do ano no hemisfério norte, as estações são as seguintes:
# 1 - Inverno (21 de dezembro a 20 de março)
# 2 - Primavera (21 de março a 20 de junho)
# 3 - Verão (21 de junho a 20 de setembro)
# 4 - Outono (21 de setembro a 20 de dezembro)
def estacao_do_ano(row):
    if (row['Mês'] == 12 and row['Dia'] >= 21) or (row['Mês'] <= 3 and row['Dia'] <= 20):
        return 'Inverno'
    elif (row['Mês'] >= 3 and row['Dia'] >= 21) or (row['Mês'] >= 4 and row['Mês'] <= 6) or (row['Mês'] <= 6 and row['Dia'] <= 20):
        return 'Primavera'
    elif (row['Mês'] >= 6 and row['Dia'] >= 21) or (row['Mês'] >= 7 and row['Mês'] <= 9) or (row['Mês'] <= 9 and row['Dia'] <= 20):
        return 'Verão'
    else:
        return 'Outono'

df['Estação do Ano'] = df.apply(estacao_do_ano, axis=1)

# Visualize o DataFrame com a nova coluna "Estação do Ano"
print(df)

#salvar arquivo tratado
df.to_csv('solar_tratado.csv', index=False)
