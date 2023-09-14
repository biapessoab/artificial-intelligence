import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('./data.csv')

X = data.iloc[:, 0:4].values
Y = data.iloc[:, 4].values

label_encoder_aparencia = LabelEncoder()
label_encoder_temperatura = LabelEncoder()
label_encoder_umidade = LabelEncoder()
label_encoder_ventando = LabelEncoder()

X[:, 0] = label_encoder_aparencia.fit_transform(X[:, 0])
X[:, 1] = label_encoder_temperatura.fit_transform(X[:, 1])
X[:, 2] = label_encoder_umidade.fit_transform(X[:, 2])
X[:, 3] = label_encoder_ventando.fit_transform(X[:, 3])

naive = GaussianNB()
naive = naive.fit(X, Y)

previsao = naive.predict([[0,1,1,1]])
print(previsao)