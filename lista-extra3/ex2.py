import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Carregar dados de treino e teste
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Colunas de interesse
cols_target = ['obscene', 'insult', 'toxic', 'severe_toxic', 'identity_hate', 'threat']

# Verificar valores faltantes em colunas numéricas
train_df.describe()

# Percentual de comentários não rotulados
unlabelled_in_all = train_df[(train_df['toxic']!=1) & (train_df['severe_toxic']!=1) & (train_df['obscene']!=1) & 
                            (train_df['threat']!=1) & (train_df['insult']!=1) & (train_df['identity_hate']!=1)]
print('Percentual de comentários não rotulados: ', len(unlabelled_in_all)/len(train_df)*100)

# Verificar se há comentários nulos
no_comment_train = train_df[train_df['comment_text'].isnull()]
len(no_comment_train)

# Verificar se há comentários nulos no conjunto de teste
no_comment_test = test_df[test_df['comment_text'].isnull()]
no_comment_test

# Informações sobre o número total de linhas nos conjuntos de treino e teste, e números para várias categorias
print('Total de linhas no conjunto de teste é {}'.format(len(test_df)))
print('Total de linhas no conjunto de treino é {}'.format(len(train_df)))
print(train_df[cols_target].sum())

# Comprimento dos textos no conjunto de treino
train_df['char_length'] = train_df['comment_text'].apply(lambda x: len(str(x)))

# Matriz de correlação entre features e targets
data = train_df[cols_target]
colormap = plt.cm.plasma
plt.figure(figsize=(7,7))
plt.title('Correlação entre features e targets', y=1.05, size=14)
sns.heatmap(data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)

# Comprimento dos textos no conjunto de teste
test_df['char_length'] = test_df['comment_text'].apply(lambda x: len(str(x)))

# Histograma do comprimento dos textos no conjunto de teste
plt.figure()
plt.hist(test_df['char_length'])
plt.title('Histograma do Comprimento do Texto no Conjunto de Teste')
plt.xlabel('Comprimento do Texto')
plt.ylabel('Frequência')
plt.show()

# Função para limpar o texto
def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    return text

# Limpar o texto nos conjuntos de treino e teste
train_df['comment_text'] = train_df['comment_text'].map(lambda com : clean_text(com))
test_df['comment_text'] = test_df['comment_text'].map(lambda com : clean_text(com))

# Remover a coluna 'char_length' do conjunto de treino
train_df = train_df.drop('char_length', axis=1)

# Separar as features e targets
X = train_df.comment_text
test_X = test_df.comment_text

print(X.shape, test_X.shape)

# Criar e ajustar o vetorizador TF-IDF
stop_words = set(stopwords.words('english'))
vect = TfidfVectorizer(max_features=5000, stop_words=stop_words)

# Criar a matriz documento-termo para o conjunto de treino
X_dtm = vect.fit_transform(X)

# Visualizar a matriz documento-termo do conjunto de treino
X_dtm

# Transformar o conjunto de teste usando o vocabulário aprendido do conjunto de treino
test_X_dtm = vect.transform(test_X)

# Visualizar a matriz documento-termo do conjunto de teste
test_X_dtm
