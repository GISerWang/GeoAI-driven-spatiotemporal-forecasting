# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CausalDCNOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, ks, ds):
        super(CausalDCNOperator, self).__init__()
        self.ps = (ks - 1) * ds
        self.ds = ds
        self.weight = nn.Parameter(torch.rand(hidden_dims, in_dims, ks))
        self.bias = nn.Parameter(torch.rand(hidden_dims))
        self.to(DEVICE)
    def forward(self, x):
        # [bs, n_timesteps, in_dims] -> [bs, in_dims, n_timesteps]
        x = x.permute(0, 2, 1)
        # [bs, hidden_dims, n_timesteps + ps]
        out = F.conv1d(x, self.weight, self.bias, 1, self.ps, self.ds)
        out = F.relu(out)
        return out[:, :, :-self.ps].contiguous()
bs = 64
n_timesteps = 10
in_dims = 3
hidden_dims = 128
ks, ds = 3, 2
train_x = torch.rand((bs, n_timesteps, in_dims))
op = CausalDCNOperator(device, in_dims, hidden_dims, ks, ds)
output = op(train_x.to(device))
print(output.size())