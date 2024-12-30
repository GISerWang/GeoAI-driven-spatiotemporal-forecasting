# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNNOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, ks):
        super(CNNOperator, self).__init__()
        self.ps = int(ks / 2)
        self.weight = nn.Parameter(torch.rand(hidden_dims, in_dims, ks))
        self.bias = nn.Parameter(torch.rand(hidden_dims))
        self.to(DEVICE)
    def forward(self, x):
        # [bs, n_timesteps, in_dims] -> [bs, in_dims, n_timesteps]
        x = x.permute(0, 2, 1)
        # [bs, in_dims, n_timesteps] -> [bs, hidden_dims, n_timesteps]
        out = F.conv1d(x, self.weight, self.bias, padding=self.ps)
        return F.relu(out)

bs = 64
n_timesteps = 10
in_dims = 3
hidden_dims = 128
ks = 5
train_x = torch.rand((bs, n_timesteps, in_dims))
op = CNNOperator(device, in_dims, hidden_dims, ks)
output = op(train_x.to(device))
print(output.size())