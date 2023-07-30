import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.read_csv('powerconsumption.csv')

df = pd.read_csv('powerconsumption.csv')

#excluir a coluna generaldiffuseflows
df = df.drop(['GeneralDiffuseFlows'], axis=1)
#excluri a coluna DiffuseFlows
df = df.drop(['DiffuseFlows'], axis=1)
#traduzir as colunas
df.columns = ['data', 'temperatura', 'umidade', 'velocidade_vento', 'consumo_ZONA_1', 'consumo_ZONA_2', 'consumo,ZONA_3']
#criar uma coluna com consumo médio das 3 zonas
df['consumo_medio'] = df[['consumo_ZONA_1', 'consumo_ZONA_2', 'consumo,ZONA_3']].mean(axis=1)
#tirar o horário da coluna data e deixar somente a data
df['data'] = pd.to_datetime(df['data']).dt.date

def mapear_estacao(df):
    primavera_inicio = pd.to_datetime('2017-03-20').date()
    primavera_fim = pd.to_datetime('2017-06-21').date()
    verao_inicio = pd.to_datetime('2017-06-21').date()
    verao_fim = pd.to_datetime('2017-09-22').date()
    outono_inicio = pd.to_datetime('2017-09-22').date()
    outono_fim = pd.to_datetime('2017-12-21').date()

    if primavera_inicio <= df < primavera_fim:
        return 'Primavera'
    elif verao_inicio <= df < verao_fim:
        return 'Verão'
    elif outono_inicio <= df < outono_fim:
        return 'Outono'
    else:
        return 'Inverno'

# Cria uma coluna com a estação do ano no DataFrame original
df['estação'] = df['data'].apply(mapear_estacao)

#colocar a coluna estação na segunda posição
cols = list(df)
cols.insert(1, cols.pop(cols.index('estação')))
df = df.loc[:, cols]

#salvar o arquivo novo
df.to_csv('powerconsumption_tratado.csv', index=False)

