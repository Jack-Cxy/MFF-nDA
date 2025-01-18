import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Trans import Trans
from utils import seed_everything
seed_everything(42)


heter_rwr = np.loadtxt(r'D:\mycode\snoRNA-disease\MSNF-MF\MSNF\heterogeneous_3_2.txt')
heter_rwr = torch.from_numpy(np.array(heter_rwr,dtype='float32'))
print(heter_rwr.shape)
d_graph_model = 357
nhead = 7
dim_feedforward = 2048
dropout = 0.3
num_layers = 3
model = Trans(d_graph_model, nhead, dim_feedforward, dropout, num_layers)

h = model(heter_rwr)
np.savetxt(r'trans_heter.txt', h.detach().numpy())
print(h.shape)