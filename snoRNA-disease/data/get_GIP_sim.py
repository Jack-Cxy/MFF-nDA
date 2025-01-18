import pandas as pd
import numpy as np
import math


# 高斯线性核相似性
def Getgauss_snoRNA(adjacentmatrix, nm):
    KM = np.zeros((nm, nm))
    gamaa = 1
    sumnormm = 0
    for i in range(nm):
        normm = np.linalg.norm(adjacentmatrix[i]) ** 2
        sumnormm = sumnormm + normm
    gamam = gamaa / (sumnormm / nm)

    for i in range(nm):
        for j in range(nm):
            KM[i, j] = math.exp(
                -gamam * (np.linalg.norm(adjacentmatrix[i] - adjacentmatrix[j]) ** 2)
            )
    return KM


def Getgauss_disease(adjacentmatrix, nd):
    KD = np.zeros((nd, nd))
    gamaa = 1
    sumnormd = 0
    for i in range(nd):
        normd = np.linalg.norm(adjacentmatrix[:, i]) ** 2
        sumnormd = sumnormd + normd
    gamad = gamaa / (sumnormd / nd)

    for i in range(nd):
        for j in range(nd):
            KD[i, j] = math.exp(
                -(
                    gamad
                    * (np.linalg.norm(adjacentmatrix[:, i] - adjacentmatrix[:, j]) ** 2)
                )
            )
    return KD


adj_df = pd.read_excel(r"adj.xlsx", index_col=0)
sno_names = list(adj_df.index)
d_names = list(adj_df.columns)
adj = adj_df.values
num_s, num_d = adj.shape

s_gip = Getgauss_snoRNA(adj, num_s)
d_gip = Getgauss_disease(adj, num_d)
print(s_gip.shape)
print(d_gip.shape)

np.savetxt(r"..\data\Similarity matrix\GIPKs.txt", s_gip)
np.savetxt(r"..\data\Similarity matrix\GIPKd.txt", d_gip)
