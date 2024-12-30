# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ConvLSTMOperator(nn.Module):
    def __init__(self, xh2f, xh2i, xh2chat, xh2o):
        super(ConvLSTMOperator, self).__init__()
        self.xh2f = xh2f
        self.xh2i = xh2i
        self.xh2chat = xh2chat
        self.xh2o = xh2o
    def forward(self, h, c, xt):
        # [bs, width, height, in_dims] -> [bs, in_dims, width, height]
        xt = xt.permute(0, 3, 1, 2)
        # [bs, in_dims + hidden_dims, width, height]
        # -> [bs, hidden_dims, width, height]
        f = torch.sigmoid(self.xh2f(torch.cat((h, xt), dim=1)))
        # [bs, in_dims + hidden_dims, width, height]
        # -> [bs, hidden_dims, width, height]
        i = torch.sigmoid(self.xh2i(torch.cat((h, xt), dim=1)))
        # [bs, in_dims + hidden_dims, width, height]
        # -> [bs, hidden_dims, width, height]
        c_hat = torch.tanh(self.xh2chat(torch.cat((h, xt), dim=1)))
        # [bs, hidden_channels, width, height]
        c = f * c + i * c_hat
        # [bs, in_dims + hidden_dims, width, height]
        # -> [bs, hidden_dims, width, height]
        o = torch.sigmoid(self.xh2o(torch.cat((h, xt), dim=1)))
        # [bs, hidden_dims, width, height]
        h = o * torch.tanh(c)
        return h, c

class ConvLSTMModel(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, out_dims, ks):
        super(ConvLSTMModel, self).__init__()
        self.DEVICE = DEVICE
        self.ps = (int(ks[0] / 2), int(ks[1] / 2))
        self.hidden_dims = hidden_dims
        self.xh2f = nn.Conv2d(in_dims + hidden_dims, hidden_dims, ks,
                              padding=self.ps)
        self.xh2i = nn.Conv2d(in_dims + hidden_dims, hidden_dims, ks,
                              padding=self.ps)
        self.xh2chat = nn.Conv2d(in_dims + hidden_dims, hidden_dims, ks,
                                 padding=self.ps)
        self.xh2o = nn.Conv2d(in_dims + hidden_dims, hidden_dims, ks,
                              padding=self.ps)
        self.op = ConvLSTMOperator(self.xh2f, self.xh2i, self.xh2chat, self.xh2o)
        self.out_layer = nn.Conv2d(hidden_dims, out_dims, ks, padding=self.ps)
        self.to(DEVICE)
    def forward(self, x):
        bs, n_timesteps, width, height, in_dims = x.size()
        h = torch.rand((bs, self.hidden_dims, width, height)).to(self.DEVICE)
        c = torch.rand((bs, self.hidden_dims, width, height)).to(self.DEVICE)
        for t in range(n_timesteps):
            h, c = self.op(h, c, x[:, t, :])
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
model = ConvLSTMModel(device, in_dims, hidden_dims, in_dims, ks)
output = model(train_x.to(device))
print(output.size())