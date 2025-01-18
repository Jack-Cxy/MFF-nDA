import pandas as pd
import numpy as np
from utils import seed_everything
seed_everything(42)


piR = np.load(r'gmodel_feat_snoR.npy')
dis = np.load(r'gmodel_feat_dis.npy')
print(piR.shape)
print(dis.shape)


ld = []
for i in range(piR.shape[0]):
    for j in range(dis.shape[0]):
        ld1 = np.hstack((piR[i],dis[j]))
        ld.append(ld1)

ld = pd.DataFrame(np.array(ld))

adj = pd.read_excel(r'D:\mycode\snoRNA-disease\data\adj.xlsx', index_col=0)
label = pd.DataFrame(np.array(adj).flatten())

feat = pd.concat([ld,label],axis=1)

dic_piR = dict(enumerate(adj.index))
dic_dis = dict(enumerate(adj.columns))


index = []
for i in range(adj.shape[0]):
    for j in range(adj.shape[1]):
        index.append(dic_piR[i]+'+'+dic_dis[j])

feat.index = index

data = pd.concat([feat[feat.iloc[:,-1]==1],
                  feat[feat.iloc[:,-1]==0].sample(frac=1, random_state=42).iloc[:780,:]],axis=0)

data = data.iloc[:, :-1]


np.save(r'D:\mycode\snoRNA-disease\data\feat\gat_feat.npy', data)

