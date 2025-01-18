import pandas as pd
import numpy as np


df = pd.read_excel('final_snoRNA-disease2.xlsx')
data = df.iloc[:, 1:3]

unique_df = data.drop_duplicates()
unique_df.to_excel('disease_doid.xlsx', index=False)


adj_matrix = pd.crosstab(df['RNA Symbol'], df['Disease Name'])

adj_matrix.to_excel('adj.xlsx')
print(adj_matrix.ne(0).sum().sum())

