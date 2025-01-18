import pandas as pd

sequence = pd.read_excel(r'snoRNA-seq.xlsx')
print(sequence.shape)
doid = pd.read_excel(r'disease_doid.xlsx')
print(doid.shape)
association = pd.read_excel(r'association.xlsx')
association = association[association['disease_name'].isin(doid.iloc[:, 0])]
association = association.drop_duplicates()
print(association.shape)
adj_matrix = pd.crosstab(association['snoRNA'], association['disease_name'])
adj_matrix.to_excel('adj.xlsx')
print(adj_matrix.shape)
print(adj_matrix.ne(0).sum().sum())
