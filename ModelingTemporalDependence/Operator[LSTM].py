# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LSTMOperator(nn.Module):
    def __init__(self, xh2f, xh2i, xh2chat, xh2o):
        super(LSTMOperator, self).__init__()
        self.xh2f = xh2f
        self.xh2i = xh2i
        self.xh2chat = xh2chat
        self.xh2o = xh2o
    def forward(self, h, c, xt):
        # [bs, in_dims + hidden_dims] -> [bs, hidden_dims]
        f = F.sigmoid(self.xh2f(torch.cat((h, xt), dim=1)))
        # [bs, in_dims + hidden_dims] -> [bs, hidden_dims]
        i = F.sigmoid(self.xh2i(torch.cat((h, xt), dim=1)))
        # [bs, in_dims + hidden_dims] -> [bs, hidden_dims]
        c_hat = F.tanh(self.xh2chat(torch.cat((h, xt), dim=1)))
        # [bs, hidden_dims]
        c = f * c + i * c_hat
        # [bs, in_dims + hidden_dims] -> [bs, hidden_dims]
        o = F.sigmoid(self.xh2o(torch.cat((h, xt), dim=1)))
        # [bs, hidden_dims]
        h = o * F.tanh(c)
        return h, c

class LSTMModel(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, out_dims):
        super(LSTMModel, self).__init__()
        self.DEVICE = DEVICE
        self.hidden_dims = hidden_dims
        self.xh2f = nn.Linear(in_dims + hidden_dims, hidden_dims)
        self.xh2i = nn.Linear(in_dims + hidden_dims, hidden_dims)
        self.xh2chat = nn.Linear(in_dims + hidden_dims, hidden_dims)
        self.xh2o = nn.Linear(in_dims + hidden_dims, hidden_dims)
        self.op = LSTMOperator(self.xh2f, self.xh2i, self.xh2chat, self.xh2o)
        self.out_layer = nn.Linear(hidden_dims, out_dims)
        self.to(DEVICE)
    def forward(self, x):
        bs, n_timesteps, in_dims = x.size()
        h = torch.rand((bs, self.hidden_dims)).to(self.DEVICE)
        c = torch.rand((bs, self.hidden_dims)).to(self.DEVICE)
        for t in range(n_timesteps):
            h, c = self.op(h, c, x[:, t, :])
        return self.out_layer(h)
bs = 64
n_timesteps = 10
in_dims = 3
hidden_dims = 128
train_x = torch.rand((bs, n_timesteps, in_dims))
model = LSTMModel(device, in_dims, hidden_dims, in_dims)
output = model(train_x.to(device))
print(output.size())