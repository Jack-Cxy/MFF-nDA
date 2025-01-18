import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from utils import min_max_normalization


class Trans(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_layers):
        super(Trans, self).__init__()
        self.atten = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation=nn.ReLU(inplace=True))
        self.encoder = nn.TransformerEncoder(self.atten, num_layers)

    def forward(self, x):
        x = self.encoder(x)
        x = min_max_normalization((x))
        return x
