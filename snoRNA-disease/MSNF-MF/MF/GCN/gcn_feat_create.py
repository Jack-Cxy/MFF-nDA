from Graph_create import get_ass,Graph_create
import torch
import torch.nn as nn
import numpy as np
from model import GCN
import pandas as pd
from utils import  seed_everything
seed_everything(42)

snoRNA_disease = get_ass(r'ass.txt')
snoRNA_feat = np.loadtxt(r'D:\mycode\snoRNA-disease\data\Similarity matrix\integrated data\snoRNA_sim_3.txt')
dis_feat = np.loadtxt(r'D:\mycode\snoRNA-disease\data\Similarity matrix\integrated data\disease_sim_2.txt')

snoRNA_feat = np.array(snoRNA_feat, dtype='float32')
dis_feat = np.array(dis_feat, dtype='float32')

snoRNA_feat = torch.from_numpy(snoRNA_feat)
dis_feat = torch.from_numpy(dis_feat)

graph, graph_h = Graph_create(snoRNA_disease, snoRNA_feat, dis_feat)

model = GCN(C_dim=307, S_dim=50, hidden_dim=128)

h = model(graph, graph_h)

np.save(r'gmodel_feat_snoR.npy', h['snoRNA'].detach().numpy())
np.save(r'gmodel_feat_dis.npy', h['disease'].detach().numpy())
