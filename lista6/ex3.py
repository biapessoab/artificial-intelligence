import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import davies_bouldin_score

import pandas as pd
base = pd.read_csv('./iris.csv')

Entrada = base.iloc[:, 0:4].values

scaler = MinMaxScaler()
Entrada = scaler.fit_transform(Entrada)

wcss = []
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=10, n_init=10) 
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
optimal_k = kl.elbow

kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10)
saida_kmeans = kmeans.fit_predict(Entrada)

dbi_score = davies_bouldin_score(Entrada, saida_kmeans)
print("Davies-Bouldin Index:", dbi_score)

cluster_means = pd.DataFrame(Entrada, columns=base.columns[0:4])
cluster_means['Cluster'] = saida_kmeans

means_by_cluster = cluster_means.groupby('Cluster').mean()
print("Cluster Means:")
print(means_by_cluster)

plt.scatter(Entrada[saida_kmeans == 0, 0], Entrada[saida_kmeans == 0, 1], s=100, c='purple', label='Iris-setosa')
plt.scatter(Entrada[saida_kmeans == 1, 0], Entrada[saida_kmeans == 1, 1], s=100, c='orange', label='Iris-versicolour')
plt.scatter(Entrada[saida_kmeans == 2, 0], Entrada[saida_kmeans == 2, 1], s=100, c='green', label='Iris-virginica')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')
plt.legend()

plt.show()
