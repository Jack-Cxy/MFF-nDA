import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from utils import seed_everything
seed_everything(42)


adj = pd.read_excel(r'D:\mycode\snoRNA-disease\data\adj.xlsx',index_col=0)
print(adj.shape)
a, b = adj.shape
trans_feat = pd.DataFrame(np.loadtxt(r'trans_heter.txt'))
print(trans_feat.shape)
sR = np.array(trans_feat.iloc[:a])
dis = np.array(trans_feat.iloc[a:])

ld = []
for i in range(sR.shape[0]):
    for j in range(dis.shape[0]):
        ld1 = np.hstack((sR[i],dis[j]))
        ld.append(ld1)

ld = np.array(ld).astype(np.float32)
label = pd.DataFrame(np.array(adj).flatten())
feat = pd.concat([pd.DataFrame(np.array(ld)), label], axis=1)
feat = pd.DataFrame(feat)
dic_piR = dict(enumerate(adj.index))
dic_dis = dict(enumerate(adj.columns))

index = []
for i in range(adj.shape[0]):
    for j in range(adj.shape[1]):
        index.append(dic_piR[i]+'+'+dic_dis[j])

feat.index = index
data = pd.concat([feat[feat.iloc[:,-1]==1],
                  feat[feat.iloc[:,-1]==0].sample(frac=1, random_state=42).iloc[:780,:]],axis=0)
data.to_csv(r'label_bal.csv')

r = pd.read_csv(r'label_bal.csv', index_col=0)
label = r.iloc[:,-1]
# print(label)
index = data.index
data = data.iloc[:,:-1]
# print(data)
# 定义全连接层
fc1 = nn.Linear(in_features=714, out_features=357)

data = np.array(data,dtype='float32')
data = torch.from_numpy(data)
h = fc1(data)
relu = nn.ReLU(inplace=True)
h = relu(h)
print(h.shape)
np.save(r'D:\mycode\snoRNA-disease\data\feat\trans_feat.npy',h.detach().numpy())
