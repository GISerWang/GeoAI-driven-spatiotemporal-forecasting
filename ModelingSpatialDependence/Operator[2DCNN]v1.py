# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNNOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, ks):
        super(CNNOperator, self).__init__()
        self.ps = (int(ks[0] / 2), int(ks[1] / 2))
        self.conv2d = nn.Conv2d(in_dims, hidden_dims, ks, padding=self.ps)
        self.to(DEVICE)
    def forward(self, x):
        # [bs, width, height, in_dims] -> [bs, in_dims, width, height]
        x = x.permute(0, 3, 1, 2)
        # [bs, hidden_dims, width, height]
        out = self.conv2d(x)
        return torch.relu(out)

bs = 64
width = 27
height = 24
in_dims = 4
hidden_dims = 128
kernel_sizes = (3, 3)
train_x = torch.rand((bs, width, height, in_dims))
op = CNNOperator(device, in_dims, hidden_dims, kernel_sizes)
output = op(train_x.to(device))
print(output.size())