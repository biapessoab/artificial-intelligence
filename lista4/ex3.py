import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

# Fazer previsões nos dados de teste
previsao = random_forest.predict([[0,1,1,0]])
print(previsao)