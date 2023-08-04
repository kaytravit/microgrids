import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv('powerconsumption_tratado.csv')


# Salvar a coluna 'data' em uma nova variável
data_column = df['data']

#dividir o dataset em 2, um para treino e outro para teste
from sklearn.model_selection import train_test_split
X = df.drop(['data' ,'consumo_medio', 'estação', 'consumo_ZONA_1', 'consumo_ZONA_2', 'consumo,ZONA_3'], axis=1)
y = df['consumo_medio']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#normalizar os dados
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#usando rfr
from sklearn.ensemble import RandomForestRegressor
model_rfr = RandomForestRegressor()
model_rfr.fit(X_train, y_train)

#fazer a previsão usando rfr
y_pred_rfr = model_rfr.predict(X_test)

# Importar a função mean_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Calcular e exibir o MAE
mae = mean_absolute_error(y_test, y_pred_rfr)
print("MAE:", mae)

# Calcular e exibir o RMSE
rmse = mean_squared_error(y_test, y_pred_rfr, squared=False)
print("RMSE:", rmse)

# Criar DataFrame com valores reais e previstos e mês correspondente
df_resultado = pd.DataFrame({'Real': y_test, 'Previsto': y_pred_rfr})

# Adicionar coluna 'Mês' ao DataFrame de resultados
df_resultado['Mês'] = pd.to_datetime(data_column).dt.month

# Plot do line plot valor previsto x valor real por mês
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_resultado, x='Mês', y='Real', label='Valor Real')
sns.lineplot(data=df_resultado, x='Mês', y='Previsto', label='Valor Previsto')
plt.xlabel('Mês')
plt.ylabel('Consumo Médio')
plt.title('Valor Previsto x Valor Real por Mês')
plt.legend()
plt.show()

# Análise de Importância de Atributos
importances = model_rfr.feature_importances_
indices = np.argsort(importances)[::-1]

# Exibir a importância de cada atributo
print("Importância dos atributos:")
for i, idx in enumerate(indices):
    print(f"{i+1}. {X.columns[idx]}: {importances[idx]}")

# Plot da importância de cada atributo
plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=X.columns[indices])
plt.xlabel('Importância')
plt.ylabel('Atributo')
plt.title('Importância dos Atributos')
plt.show()

