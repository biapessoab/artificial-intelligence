from chefboost import Chefboost as chef
import pandas as pd

df = pd.read_csv("data.csv")
config = {'algorithm': 'ID3'}
model = chef.fit(df, config = config, target_label = 'Decision')
