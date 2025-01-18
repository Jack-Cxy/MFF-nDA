from Graph_create import get_ass,Graph_create
import torch
import torch.nn as nn
import numpy as np
from model import GAT
import pandas as pd
from utils import seed_everything
seed_everything(42)


snoRNA_disease = get_ass(r'ass.txt')
snoRNA_feat = np.loadtxt(r'D:\mycode\snoRNA-disease\data\Similarity matrix\integrated data\snoRNA_sim_3.txt')
dis_feat = np.loadtxt(r'D:\mycode\snoRNA-disease\data\Similarity matrix\integrated data\disease_sim_2.txt')

feature_dim = 128

snoRNA_fc = nn.Linear(307, feature_dim)
dis_fc = nn.Linear(50, feature_dim)

snoRNA_feat = snoRNA_fc(torch.tensor(snoRNA_feat, dtype=torch.float32))
dis_feat = dis_fc(torch.tensor(dis_feat, dtype=torch.float32))

snoRNA_feat = snoRNA_feat.detach().numpy().astype('float32')
dis_feat = dis_feat.detach().numpy().astype('float32')

snoRNA_feat = torch.from_numpy(snoRNA_feat)
dis_feat = torch.from_numpy(dis_feat)

graph, graph_h = Graph_create(snoRNA_disease, snoRNA_feat, dis_feat)

model = GAT(C_dim=feature_dim, S_dim=feature_dim, hidden_dim=128, num_heads=1)

h = model(graph, graph_h)
snoRNA_feat_ = h['snoRNA'].detach().numpy().squeeze()
disease_feat_ = h['disease'].detach().numpy().squeeze()

snoRNA_fc_ = nn.Linear(feature_dim,307)
dis_fc_ = nn.Linear( feature_dim, 50)

snoRNA_feat_ = snoRNA_fc_(torch.tensor(snoRNA_feat_, dtype=torch.float32))
dis_feat_ = dis_fc_(torch.tensor(disease_feat_, dtype=torch.float32))

snoRNA_feat_ = snoRNA_feat_.detach().numpy().astype('float32')
dis_feat_ = dis_feat_.detach().numpy().astype('float32')

np.save(r'gmodel_feat_snoR.npy',snoRNA_feat_)
np.save(r'gmodel_feat_dis.npy',dis_feat_)
