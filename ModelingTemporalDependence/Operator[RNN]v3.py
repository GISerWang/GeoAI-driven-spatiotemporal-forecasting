# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class RNNOperator(nn.Module):
    def __init__(self, w, b):
        super(RNNOperator, self).__init__()
        self.w = w
        self.b = b
    def forward(self, h, xt):
        # [batch_size, hidden_dims]
        return torch.tanh(F.linear(torch.cat((h, xt), dim=1), self.w, self.b))

class RNNModel(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, out_dims):
        super(RNNModel, self).__init__()
        self.DEVICE = DEVICE
        self.hidden_dims = hidden_dims
        self.w = nn.Parameter(torch.rand(hidden_dims, in_dims + hidden_dims))
        self.b = nn.Parameter(torch.rand(hidden_dims))
        self.op = RNNOperator(self.w, self.b)
        self.out_layer = nn.Linear(hidden_dims, out_dims)
        self.to(DEVICE)
    def forward(self, x):
        bs, n_timesteps, in_dims = x.size()
        h = torch.rand((bs, self.hidden_dims)).to(self.DEVICE)
        for t in range(n_timesteps):
            h = self.op(h, x[:, t, :])
        return self.out_layer(h)
bs = 64
n_timesteps = 10
in_dims = 3
hidden_dims = 128
train_x = torch.rand((bs, n_timesteps, in_dims))
model = RNNModel(device, in_dims, hidden_dims, in_dims)
output = model(train_x.to(device))
print(output.size())