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

# print itemsets with their respective supports
for rule in rules:
    items = [item for item in rule.items]
    support = rule.support
    print(f"Itens: {items}, Suporte: {support:.4f}")
