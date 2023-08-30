import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree

# Carregando o csv
data = pd.read_csv('restaurante.csv', usecols=['Instancia','Bar','Sex/Sab','Fome','Cliente','Preco','Chuva','Res','Tipo','Tempo','Conclusao'])

# Declarando os valores de cada posição + o rótulo
X_prev = data.iloc[:, 0:10].values
label = data.iloc[:, 10].values

# Tratando dados categóricos
label_encoder = LabelEncoder()

label_encoder_Alternativo = LabelEncoder()
label_encoder_Bar = LabelEncoder()
label_encoder_SexSab = LabelEncoder()
label_encoder_fome = LabelEncoder()
label_encoder_chuva = LabelEncoder()
label_encoder_Res = LabelEncoder()

X_prev[:,0] = label_encoder_Alternativo.fit_transform(X_prev[:,1])
X_prev[:,1] = label_encoder_Bar.fit_transform(X_prev[:,2])
X_prev[:,2] = label_encoder_SexSab.fit_transform(X_prev[:,3])
X_prev[:,3] = label_encoder_fome.fit_transform(X_prev[:,4])
X_prev[:,4] = label_encoder_chuva.fit_transform(X_prev[:,5])
X_prev[:,5] = label_encoder_Res.fit_transform(X_prev[:,6])
X_prev[:,6] = label_encoder_Alternativo.fit_transform(X_prev[:,7])
X_prev[:,7] = label_encoder_Bar.fit_transform(X_prev[:,8])
X_prev[:,9] = label_encoder_SexSab.fit_transform(X_prev[:,9])

# print(X_prev)

# One Hot Enconder -> binarizar atributos não ordinais

onehotencoder_data = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [8])], remainder='passthrough')

X_prev= onehotencoder_data.fit_transform(X_prev)

# print(X_prev)

# Holdout
X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, label, test_size = 0.20, random_state = 23)

# Decision tree
model = DecisionTreeClassifier(criterion='entropy')
Y = model.fit(X_treino, y_treino)
prevision = model.predict(X_teste)
# print(y_teste)
# print(accuracy_score(y_teste,prevision))
confusion_matrix(y_teste, prevision)

cm = ConfusionMatrix(model)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)

# print(classification_report(y_teste, prevision))

# Plotando a árvore
tree.plot_tree(Y)
plt.show()