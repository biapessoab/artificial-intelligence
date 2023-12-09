import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight

# Carregando dados de treinamento e teste
treino = pd.read_csv('ReutersGrain-train.csv', escapechar='\\', quotechar="'")
teste = pd.read_csv('ReutersGrain-test.csv', escapechar='\\', quotechar="'")

# Pré-processamento de texto
treino['Text'] = treino['Text'].str.replace('\n', ' ').str.lower()
teste['Text'] = teste['Text'].str.replace('\n', ' ').str.lower()

# Divisão dos dados em conjunto de treinamento e teste
X_train, y_train = treino['Text'], treino['class-att']
X_test, y_test = teste['Text'], teste['class-att']

# Vetorização do texto usando TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Verificando distribuição de rótulos
print("Distribuição de rótulos no conjunto de treinamento:")
print(y_train.value_counts())

print("\nDistribuição de rótulos no conjunto de teste:")
print(y_test.value_counts())

# Ajuste de pesos de classe
class_weights = class_weight.compute_class_weight('balanced', classes=[0, 1], y=y_train)
clf = MultinomialNB(class_prior=class_weights)

# Treinamento do modelo Naive Bayes
clf.fit(X_train_tfidf, y_train)

# Fazendo previsões no conjunto de teste
y_pred = clf.predict(X_test_tfidf)

# Mostrando a classificação de cada texto
for i, (text, true_label, predicted_label) in enumerate(zip(X_test, y_test, y_pred)):
    print(f"Texto {i+1}:")
    print("Rótulo verdadeiro:", true_label)
    print("Rótulo previsto:", predicted_label)
    print("\n---\n")

# Avaliação do modelo
print("Acurácia:", accuracy_score(y_test, y_pred))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred))
