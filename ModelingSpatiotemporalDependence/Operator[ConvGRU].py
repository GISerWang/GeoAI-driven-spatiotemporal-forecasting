# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ConvGRUOperator(nn.Module):
    def __init__(self, inh2r, inh2z, in2n, h2n):
        super(ConvGRUOperator, self).__init__()
        self.inh2r = inh2r
        self.inh2z = inh2z
        self.in2n = in2n
        self.h2n = h2n
    def forward(self, h, xt):
        # [bs, width, height, in_dims]
        # -> [bs, in_dims, width, height]
        xt = xt.permute(0, 3, 1, 2)
        # [bs, in_dims + hidden_dims, width, height]
        # -> [bs, hidden_dims, width, height]
        r = torch.sigmoid(self.inh2r(torch.cat((h, xt), dim=1)))
        # [bs, in_dims + hidden_dims, width, height]
        # -> [bs, hidden_dims, width, height]
        z = torch.sigmoid(self.inh2z(torch.cat((h, xt), dim=1)))
        # [bs, hidden_dims]
        n = torch.tanh(self.in2n(xt) + r * self.h2n(h))
        return (1 - z) * n + z * h

class ConvGRUModel(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, out_dims, ks):
        super(ConvGRUModel, self).__init__()
        self.DEVICE = DEVICE
        self.ps = (int(ks[0] / 2), int(ks[1] / 2))
        self.hidden_dims = hidden_dims
        self.inh2r = nn.Conv2d(in_dims + hidden_dims, hidden_dims, ks,
                               padding=self.ps)
        self.inh2z = nn.Conv2d(in_dims + hidden_dims, hidden_dims, ks,
                               padding=self.ps)
        self.in2n = nn.Conv2d(in_dims, hidden_dims, ks, padding=self.ps)
        self.h2n = nn.Conv2d(hidden_dims, hidden_dims, ks, padding=self.ps)
        self.op = ConvGRUOperator(self.inh2r, self.inh2z, self.in2n, self.h2n)
        self.out_layer = nn.Conv2d(hidden_dims, out_dims, ks, padding=self.ps)
        self.to(DEVICE)
    def forward(self, x):
        bs, n_timesteps, width, height, in_dims = x.size()
        h = torch.rand((bs, self.hidden_dims, width, height)).to(self.DEVICE)
        for t in range(n_timesteps):
            h = self.op(h, x[:, t, :, :, :])
        # [bs, in_dims, width, height] -> [bs, width, height, in_dims]
        return self.out_layer(h).permute(0, 2, 3, 1)
bs = 64
n_timesteps = 10
in_dims = 4
hidden_dims = 128
width = 27
height = 24
ks = (3, 3)
train_x = torch.rand((bs, n_timesteps, width, height, in_dims))
model = ConvGRUModel(device, in_dims, hidden_dims, in_dims, ks)
output = model(train_x.to(device))
print(output.size())