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

base = pd.read_csv('restaurante.csv', usecols=['Instancia','Bar','Sex/Sab','Fome','Cliente','Preco','Chuva','Res','Tipo','Tempo','Conclusao']) # carregando a base csv

# sns.countplot(x = base['Conclusao'])

X_prev = base.iloc[:, 0:10].values
X_prev_label = base.iloc[:, 0:10]
y_classe = base.iloc[:, 10].values


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

onehotencoder_restaurante = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [8])], remainder='passthrough')

X_prev= onehotencoder_restaurante.fit_transform(X_prev)

# print(X_prev)

# Holdout
X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size = 0.20, random_state = 23)

# Decision tree

modelo = DecisionTreeClassifier(criterion='entropy')
Y = modelo.fit(X_treino, y_treino)
previsoes = modelo.predict(X_teste)
print(y_teste)
print(accuracy_score(y_teste,previsoes))
confusion_matrix(y_teste, previsoes)