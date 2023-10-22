import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from kneed import DataGenerator, KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Carregar o conjunto de dados Iris
base = pd.read_csv('./iris.csv')

# Selecionar os atributos de entrada
Entrada = base.iloc[:, 0:4].values

# Normalizar os dados
scaler = MinMaxScaler()
Entrada = scaler.fit_transform(Entrada)

# Determinar o número ideal de clusters usando o Método do Cotovelo
wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=10)
    kmeans.fit(Entrada)
    wcss.append(kmeans.inertia_)

plt.style.use("fivethirtyeight")
plt.plot(range(2, 11), wcss)
plt.xticks(range(2, 11))
plt.title('The elbow method')
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

kl = KneeLocator(range(2, 11), wcss, curve="convex", direction="decreasing")
numero_de_clusters = kl.elbow

# Executar o K-Means com o número ideal de clusters
kmeans = KMeans(n_clusters=numero_de_clusters, random_state=0)
saida_kmeans = kmeans.fit_predict(Entrada)

# Visualizar os clusters
fig = px.scatter(x=Entrada[:, 0], y=Entrada[:, 1], color=saida_kmeans, labels={'color': 'Cluster'})
fig.show()

# Agora, para destacar as instâncias agrupadas incorretamente:
# Comparar os clusters atribuídos pelo K-Means com as classes reais
classes_reais = base['classe'].values

# Encontrar as instâncias agrupadas incorretamente
incorretamente_agrupadas = Entrada[saida_kmeans != classes_reais]

# Criar um gráfico de dispersão para destacar as instâncias incorretamente agrupadas
fig_incorretas = px.scatter(x=Entrada[:, 0], y=Entrada[:, 1], color=saida_kmeans,
                            labels={'color': 'Cluster'}, title='Instâncias Incorretamente Agrupadas')
fig_incorretas.add_trace(go.Scatter(x=incorretamente_agrupadas[:, 0], y=incorretamente_agrupadas[:, 1],
                                    mode='markers', marker=dict(size=10, color='red'),
                                    name='Incorretamente Agrupadas'))
fig_incorretas.show()
