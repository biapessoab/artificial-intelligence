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
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn import tree

data = pd.read_csv('./heart.csv')

X = data.iloc[:, 0:11].values
Y = data.iloc[:, 11].values

label_encoder_age = LabelEncoder()
label_encoder_sex = LabelEncoder()
label_encoder_chestPainType = LabelEncoder()
label_encoder_restingBP = LabelEncoder()
label_encoder_cholesterol = LabelEncoder()
label_encoder_fastingBS = LabelEncoder()
label_encoder_restingECG = LabelEncoder()
label_encoder_maxHR = LabelEncoder()
label_encoder_exerciseAngina = LabelEncoder()
label_encoder_oldPeak = LabelEncoder()
label_encoder_stSlope = LabelEncoder()

X[:, 0] = label_encoder_age.fit_transform(X[:, 0])
X[:, 1] = label_encoder_sex.fit_transform(X[:, 1])
X[:, 2] = label_encoder_chestPainType.fit_transform(X[:, 2])
X[:, 3] = label_encoder_restingBP.fit_transform(X[:, 3])
X[:, 4] = label_encoder_cholesterol.fit_transform(X[:, 4])
X[:, 5] = label_encoder_fastingBS.fit_transform(X[:, 5])
X[:, 6] = label_encoder_restingECG.fit_transform(X[:, 6])
X[:, 7] = label_encoder_maxHR.fit_transform(X[:, 7])
X[:, 8] = label_encoder_exerciseAngina.fit_transform(X[:, 8])
X[:, 9] = label_encoder_oldPeak.fit_transform(X[:, 9])
X[:, 10] = label_encoder_stSlope.fit_transform(X[:, 10])

onehotencoder_data = ColumnTransformer(transformers=[('OneHot', OneHotEncoder(), [8])], remainder='passthrough')

X_prev= onehotencoder_data.fit_transform(X)

# Holdout
X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size = 0.20, random_state = 23)

# Decision tree
model = DecisionTreeClassifier(criterion='entropy')
Y = model.fit(X_treino, y_treino)
prevision = model.predict(X_teste)

confusion_matrix(y_teste, prevision)

cm = ConfusionMatrix(model)
cm.fit(X_treino, y_treino)
cm.score(X_teste, y_teste)


# Plotando a árvore
tree.plot_tree(Y)
plt.show()