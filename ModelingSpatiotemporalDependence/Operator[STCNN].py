# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class STCNNOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, ks):
        super(STCNNOperator, self).__init__()
        self.ps = (ks[0] - 1, int(ks[1] / 2), int(ks[1] / 2))
        self.conv3d = nn.Conv3d(in_dims, hidden_dims, ks, padding=self.ps)
        self.to(DEVICE)
    def forward(self, x):
        # [bs, n_timesteps, width, height, in_dims]
        # -> [bs, in_dims, n_timesteps, width, height]
        x = x.permute(0, 4, 1, 2, 3)
        # [bs, hidden_dims, n_timesteps + ps[0], width, height]
        out = self.conv3d(x)
        # [batch_size, hidden_dims, n_timesteps, width, height]
        out = out[:, :, :-self.ps[0], :, :]
        return torch.relu(out)

bs = 64
n_timesteps = 10
in_dims = 4
hidden_dims = 128
width = 27
height = 24
ks = (3, 3, 3)
train_x = torch.rand((bs, n_timesteps, width, height, in_dims))
op = STCNNOperator(device, in_dims, hidden_dims, ks)
output = op(train_x.to(device))
print(output.size())