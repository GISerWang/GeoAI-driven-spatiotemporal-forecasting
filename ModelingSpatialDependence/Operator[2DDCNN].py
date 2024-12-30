# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DCNNOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, ks, ds):
        super(DCNNOperator, self).__init__()
        self.ds = ds
        self.ps = (int((ks[0] - 1) * ds[0] / 2), int((ks[1] - 1) * ds[1] / 2))
        self.weight = nn.Parameter(torch.rand(hidden_dims, in_dims, ks[0], ks[1]))
        self.bias = nn.Parameter(torch.rand(hidden_dims))
        self.to(DEVICE)
    def forward(self, x):
        # [bs, width, height, in_dims] -> [bs, in_dims, width, height]
        x = x.permute(0, 3, 1, 2)
        # [bs, hidden_dims, width, height]
        out = F.conv2d(x, self.weight, self.bias,
                       padding=self.ps,
                       dilation=self.ds)
        return torch.relu(out)

bs = 64
width = 27
height = 24
in_dims = 4
hidden_dims = 128
ks, ds = (3, 3), (1, 3)
train_x = torch.rand((bs, width, height, in_dims))
op = DCNNOperator(device, in_dims, hidden_dims, ks, ds)
output = op(train_x.to(device))
print(output.size())