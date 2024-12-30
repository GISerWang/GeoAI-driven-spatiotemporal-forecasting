# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GRUOperator(nn.Module):
    def __init__(self, inh2r, inh2z, in2n, h2n):
        super(GRUOperator, self).__init__()
        self.inh2r = inh2r
        self.inh2z = inh2z
        self.in2n = in2n
        self.h2n = h2n
    def forward(self, h, xt):
        # [bs, in_dims + hidden_dims] -> [bs, hidden_dims]
        r = F.sigmoid(self.inh2r(torch.cat((h, xt), dim=1)))
        # [bs, in_dims + hidden_dims] -> [bs, hidden_dims]
        z = F.sigmoid(self.inh2z(torch.cat((h, xt), dim=1)))
        # [bs, hidden_dims]
        n = F.tanh(self.in2n(xt) + r * self.h2n(h))
        return (1 - z) * n + z * h

class GRUModel(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, out_dims):
        super(GRUModel, self).__init__()
        self.DEVICE = DEVICE
        self.hidden_dims = hidden_dims
        self.inh2r = nn.Linear(in_dims + hidden_dims, hidden_dims)
        self.inh2z = nn.Linear(in_dims + hidden_dims, hidden_dims)
        self.in2n = nn.Linear(in_dims, hidden_dims)
        self.h2n = nn.Linear(hidden_dims, hidden_dims)
        self.op = GRUOperator(self.inh2r, self.inh2z, self.in2n, self.h2n)
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
model = GRUModel(device, in_dims, hidden_dims, in_dims)
output = model(train_x.to(device))
print(output.size())