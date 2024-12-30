# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNNOperator(nn.Module):
    def __init__(self, DEVICE, in_dim, hidden_dims, ks):
        super(CNNOperator, self).__init__()
        ps = int(ks / 2)
        self.conv1d = nn.Conv1d(in_dim, hidden_dims, kernel_size=ks, padding=ps)
        self.to(DEVICE)
    def forward(self, x):
        # [bs, n_timesteps, in_dims] -> [bs, in_dims, n_timesteps]
        x = x.permute(0, 2, 1)
        # [bs, hidden_dims, n_timesteps]
        return F.relu(self.conv1d(x))

bs = 64
n_timesteps = 10
in_dims = 3
hidden_dims = 128
ks = 3
train_x = torch.rand((bs, n_timesteps, in_dims))
op = CNNOperator(device, in_dims, hidden_dims, ks)
output = op(train_x.to(device))
print(output.size())