# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class RNNOperator(nn.Module):
    def __init__(self, liner):
        super(RNNOperator, self).__init__()
        self.liner = liner
    def forward(self, h, xt):
        # [batch_size, hidden_dims]
        return torch.tanh(self.liner(torch.cat((h, xt), dim=1)))
class RNNModel(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, out_channels):
        super(RNNModel, self).__init__()
        self.DEVICE = DEVICE
        self.hidden_dims = hidden_dims
        self.liner = nn.Linear(in_dims + hidden_dims, hidden_dims)
        self.op = RNNOperator(self.liner)
        self.out_layer = nn.Linear(hidden_dims, out_channels)
        self.to(DEVICE)
    def forward(self, x):
        bs, n_timesteps, in_dims = x.size()
        h = torch.rand((bs, self.hidden_dims)).to(self.DEVICE)
        for t in range(n_timesteps):
            h = self.op(h, x[:, t, :])
        return self.out_layer(h)
bs = 64
n_timesteps = 10
in_channels = 3
hidden_dims = 128
train_x = torch.rand((bs, n_timesteps, in_channels))
model = RNNModel(device, in_channels, hidden_dims, in_channels)
output = model(train_x.to(device))
print(output.size())