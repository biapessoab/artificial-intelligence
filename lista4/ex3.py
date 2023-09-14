import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

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

# GridSearch
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=random_forest, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, Y_train)

Y_pred_grid = grid_search.best_estimator_.predict(X_test)

# Random Search
param_dist = {
    'n_estimators': [int(x) for x in np.linspace(start=50, stop=200, num=10)],
    'max_depth': [None] + list(np.random.randint(10, 50, size=10)),
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 8]
}

random_search = RandomizedSearchCV(estimator=random_forest, param_distributions=param_dist, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1)
random_search.fit(X_train, Y_train)

# Resultados

print("Melhores hiperparâmetros com GridSearch: " + str(grid_search.best_params_))
print("Melhores Hiperparâmetros com RandomSearch: " + str(random_search.best_params_))

Y_pred_random = random_search.best_estimator_.predict(X_test)

# Avaliação dos modelos
accuracy_grid = accuracy_score(Y_test, Y_pred_grid)
accuracy_random = accuracy_score(Y_test, Y_pred_random)

print("Classificação com GridSearch:")
print(classification_report(Y_test, Y_pred_grid))
print("Acurácia com GridSearch:", accuracy_grid)

print("Classificação com RandomSearch:")
print(classification_report(Y_test, Y_pred_random))
print("Acurácia com RandomSearch:", accuracy_random)