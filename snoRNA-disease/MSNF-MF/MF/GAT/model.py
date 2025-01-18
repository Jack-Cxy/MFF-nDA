import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import dgl.nn.pytorch as dglnn


class GAT(nn.Module):
    def __init__(self, C_dim, S_dim, hidden_dim, num_heads):
        super(GAT, self).__init__()
        self.C_dim = C_dim
        self.S_dim = S_dim
        self.hidden_dim = hidden_dim

        self.HeteroConv1 = dglnn.HeteroGraphConv({
            'p_d': GATConv(self.C_dim, self.hidden_dim, num_heads=num_heads, activation=F.relu,allow_zero_in_degree=True),
            'd_p': GATConv(self.S_dim, self.hidden_dim, num_heads=num_heads, activation=F.relu,allow_zero_in_degree=True)
        },
        aggregate='mean')

        self.HeteroConv2 = dglnn.HeteroGraphConv({
            'p_d': GATConv(self.hidden_dim * num_heads, self.hidden_dim, activation=F.relu,num_heads=num_heads, allow_zero_in_degree=True),
            'd_p': GATConv(self.hidden_dim * num_heads, self.hidden_dim, activation=F.relu,num_heads=num_heads, allow_zero_in_degree=True)
        },
        aggregate='mean')

        self.HeteroConv3 = dglnn.HeteroGraphConv({
            'p_d': GATConv(self.hidden_dim * num_heads, self.S_dim, activation=F.relu,num_heads=num_heads, allow_zero_in_degree=True),
            'd_p': GATConv(self.hidden_dim * num_heads, self.C_dim, activation=F.relu,num_heads=num_heads, allow_zero_in_degree=True)
        },
        aggregate='mean')

    def forward(self, g, h):
        h1 = self.HeteroConv1(g, h)
        h2 = self.HeteroConv2(g, h1)
        h3 = self.HeteroConv3(g, h2)
        return h3
