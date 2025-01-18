import pandas as pd
import numpy as np
from utils import seed_everything
seed_everything(42)


snoRNA_sim1 = np.loadtxt(r"D:\mycode\snoRNA-disease\data\Similarity matrix\seq_sim.txt")
print(snoRNA_sim1.shape)
snoRNA_sim2 = np.loadtxt(r"D:\mycode\snoRNA-disease\data\Similarity matrix\Func_sim.txt")
print(snoRNA_sim2.shape)
GIP_s_sim = np.loadtxt(r"D:\mycode\snoRNA-disease\data\Similarity matrix\GIPKs.txt")
print(GIP_s_sim.shape)
GIP_d_sim = np.loadtxt(r"D:\mycode\snoRNA-disease\data\Similarity matrix\GIPKd.txt")
print(GIP_d_sim.shape)
disease_sim1 = np.loadtxt(r"D:\mycode\snoRNA-disease\data\Similarity matrix\doid_sim.txt")
print(disease_sim1.shape)

Ss = (snoRNA_sim1 + snoRNA_sim2 + GIP_s_sim) / 3.0
Ds = (disease_sim1 + GIP_d_sim) / 2.0

np.savetxt(r"D:\mycode\snoRNA-disease\data\Similarity matrix\integrated data\snoRNA_sim_3.txt", Ss)
np.savetxt(r"D:\mycode\snoRNA-disease\data\Similarity matrix\integrated data\disease_sim_2.txt",Ds)
