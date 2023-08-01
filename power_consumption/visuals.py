import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv('powerconsumption_tratado.csv')

#grafico de consumo médio por dia
sns.lineplot(x='data', y='consumo_medio', data=df)
plt.show()

#grafico de consumo médio por estação em linhas num mesmo grafico
sns.lineplot(x='data', y='consumo_medio', hue='estação', data=df)
plt.show()  

#grafico de correlação
sns.heatmap(df.corr(), annot=True)
plt.show()

