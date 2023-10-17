import pandas as pd
from apyori import apriori

# read the CSV file
with open('./MercadoSim.csv', 'r') as file:
    lines = file.readlines()

# transform the dataset into a list of lists, considering only named items
transactions = []
for line in lines:
    items = [item.strip() for item in line.split(';') if item.strip()]
    transactions.append(items)

# apriori algorithm
rules = apriori(transactions, min_support=0.3, min_confidence=0.8)
saida = list(rules)

antecedent = []
consequent = []
support = []
trust = []
lift = []

# table with itemsets, support, trust and lift
for resultado in saida:
  s = resultado[1]
  result_rules = resultado[2]
  for result_rule in result_rules:
    a = list(result_rule[0])
    b = list(result_rule[1])
    c = result_rule[2]
    l = result_rule[3]
    if 'nan' in a or 'nan' in b: continue
    if len(a) == 0 or len(b) == 0: continue
    antecedent.append(a)
    consequent.append(b)
    support.append(s)
    trust.append(c)
    lift.append(l)
    RegrasFinais = pd.DataFrame({'antecedent': antecedent, 'consequent': consequent, 'support': support, 'trust': trust, 'lift': lift})

for i in range(len(antecedent)):
    print('Quem não leva ' + str(antecedent[i]) + ' leva ' + str(consequent[i]))

print(RegrasFinais)