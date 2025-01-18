import numpy as np
import pandas as pd
from utils import seed_everything
seed_everything(42)


adj = pd.read_excel(r'D:\mycode\snoRNA-disease\data\adj.xlsx', index_col=0)
print(adj.shape)
snoRNA_sim = np.loadtxt(r'D:\mycode\snoRNA-disease\data\Similarity matrix\integrated data\snoRNA_sim_3.txt')
print(snoRNA_sim.shape)
disease_sim = np.loadtxt(r'D:\mycode\snoRNA-disease\data\Similarity matrix\integrated data\disease_sim_2.txt')
print(disease_sim.shape)

snoRNA_size = snoRNA_sim.shape[0]
disease_size = disease_sim.shape[0]

combined_matrix = np.zeros((snoRNA_size + disease_size , snoRNA_size + disease_size ))

combined_matrix[:snoRNA_size, :snoRNA_size] = snoRNA_sim
combined_matrix[snoRNA_size:, snoRNA_size:] = disease_sim
combined_matrix[:snoRNA_size, snoRNA_size:] = adj.values
combined_matrix[snoRNA_size:, :snoRNA_size] = adj.values.T


np.savetxt(r"heterogeneous_3_2.txt", combined_matrix)
print(combined_matrix)