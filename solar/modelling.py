import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


df = pd.read_csv('solar_tratado.csv')
df['Data'] = pd.to_datetime(df['Data'])  
df.sort_values(by='Data', ascending=True, inplace=True) 

# Divisão treino e teste
treino = df[df['Data'] < '2016-12-01']
teste = df[df['Data'] >= '2016-12-01']

# Criar o gráfico
plt.figure(figsize=(10, 6))
plt.plot(treino['Data'], treino['Radiation'], label='Treino', color='blue')
plt.plot(teste['Data'], teste['Radiation'], label='Teste', color='red')

plt.xlabel('Data')
plt.ylabel('Radiação')
plt.title('Divisão de Treino e Teste em Comparação com a Radiação')
plt.legend()
plt.grid(True)

# Rotacionar as datas no eixo x para facilitar a leitura (opcional)
plt.xticks(rotation=45)

plt.show()

# Dividir as variáveis de treino e teste
X_treino = treino.drop(['Data', 'Radiation', 'Time', 'UNIXTime', 'TimeSunRise', 'TimeSunSet', 'Estação do Ano', 'Dia', 'Mês', 'Ano'], axis=1)
y_treino = treino['Radiation']

X_teste = teste.drop(['Data', 'Radiation', 'Time', 'UNIXTime', 'TimeSunRise', 'TimeSunSet', 'Estação do Ano', 'Dia', 'Mês', 'Ano'], axis=1)
y_teste = teste['Radiation']

print(X_treino.shape, y_treino.shape, X_teste.shape, y_teste.shape)

total_linhas = len(treino) + len(teste)
porcentagem_treino = len(treino) / total_linhas * 100
porcentagem_teste = len(teste) / total_linhas * 100

print(f"Porcentagem de dados de treino: {porcentagem_treino:.2f}%")
print(f"Porcentagem de dados de teste: {porcentagem_teste:.2f}%")

# Normalizar os dados
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_treino[:, :5] = scaler.fit_transform(X_treino[:, :5])
X_teste[:, :5] = scaler.transform(X_teste[:, :5])

# Convertendo novamente para DataFrame após a normalização
X_treino = pd.DataFrame(X_treino, columns=X_treino.columns)
X_teste = pd.DataFrame(X_teste, columns=X_teste.columns)

# Treinar o modelo

modelo_xgb = xgb.XGBRegressor(n_estimators = 1000)
modelo_xgb.fit(X_treino, y_treino, eval_set = [(X_treino, y_treino), (X_teste, y_teste)], verbose=False)

# Avaliar o modelo
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

y_pred = modelo_xgb.predict(X_teste)

# Plot do gráfico real x predito
plt.figure(figsize=(10, 6))
plt.plot(y_teste, label='Real', color='blue')
plt.plot(y_pred, label='Previsto', color='red')
plt.show()

# Plotar a importância dos recursos
importances = modelo_xgb.feature_importances_
indices = np.argsort(importances)[::-1]

# Exibir a importância de cada atributo
print("Importância dos atributos:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {X_treino.columns[idx]}: {importances[idx]}")

# Plot da importância de cada atributo
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X_treino.columns[indices])
plt.xlabel('Importância')
plt.ylabel('Atributo')
plt.title('Importância dos Atributos')
plt.show()

