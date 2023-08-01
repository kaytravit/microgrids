import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('solar_tratado.csv')

#grafico de correlação
sns.heatmap(df.corr(), annot=True)
plt.show()

#grafico entre temperatura e radiação solar
sns.lineplot(x='Temperature', y='Radiation', data=df)
plt.show()

#grafico entre umidade e radiação solar
sns.lineplot(x='Humidity', y='Radiation', data=df)
plt.show()

#grafico de linha entre radiação solar e data
sns.lineplot(x='Data', y='Radiation', data=df)
plt.show()

#grafico entre estação do ano e radiação solar
sns.lineplot(x='Estação do Ano', y='Radiation', data=df)   
plt.show()

