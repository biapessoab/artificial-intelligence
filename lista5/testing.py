import pandas as pd
from apyori import apriori
#encoding = "cp1252", encoding='utf-8', encoding='latin1'etc
base= pd.read_csv('/content/sample_data/pasta de dente python SIM.csv', ';', encoding='cp1252', header=None)
#base= pd.read_csv('/content/sample_data/pasta de dente python SIM e NÃO.csv', ';', encoding='cp1252')
#base= pd.read_csv('/content/sample_data/pasta de dente python SIM.csv', ';', encoding='cp1252',header=None)
base
base.shape