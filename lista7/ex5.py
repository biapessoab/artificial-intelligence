import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np

# Carregar seus dados
data = pd.read_csv("breast-cancer.csv")

# Pré-processamento
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

colunas_categoricas = ['menopause', 'node-caps', 'breast', 'breast-quad', 'Class']
data_categorico = data[colunas_categoricas]

encoder = OneHotEncoder(sparse_output=False, drop='first')
data_encoded = encoder.fit_transform(data_categorico)
data_encoded_df = pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(colunas_categoricas))

data = data.drop(columns=colunas_categoricas)
data = pd.concat([data, data_encoded_df], axis=1)

scaler = MinMaxScaler()
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Balanceamento com SMOTE
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Divisão dos dados em treinamento e teste após o balanceamento
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Criação do modelo de rede neural
modelo = tf.keras.Sequential()
modelo.add(tf.keras.layers.Dense(64, activation="relu", input_dim=X_train.shape[1]))
modelo.add(tf.keras.layers.Dense(32, activation="relu"))
modelo.add(tf.keras.layers.Dense(1, activation="sigmoid"))

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Treinamento do modelo
modelo.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

# Avaliação do modelo no conjunto de teste
resultado = modelo.evaluate(X_test, y_test)
print(f"Acurácia no conjunto de teste: {resultado[1]:.2f}")
