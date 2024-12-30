# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CNNOperator(nn.Module):
    def __init__(self, DEVICE, in_channels, hidden_channels, kernel_sizes, stride_sizes):
        super(CNNOperator, self).__init__()
        self.padding_sizes = (int(kernel_sizes[0] / 2), int(kernel_sizes[1] / 2), int(kernel_sizes[2] / 2))
        self.stride_sizes = stride_sizes
        self.weight = nn.Parameter(torch.rand(hidden_channels, in_channels, kernel_sizes[0], kernel_sizes[1], kernel_sizes[2]))
        self.bias = nn.Parameter(torch.rand(hidden_channels))
        self.to(DEVICE)
    def forward(self, x):
        # [batch_size, num_of_timesteps, width, height, in_channels] -> [batch_size, in_channels, num_of_timesteps, width, height]
        x = x.permute(0, 4, 1, 2, 3)
        # [batch_size, hidden_channels, num_of_timesteps, width, height]
        out = F.conv3d(x, self.weight, self.bias, stride=self.stride_sizes, padding=self.padding_sizes)
        return torch.relu(out)

batch_size = 64
num_of_timesteps = 10
in_channels = 1
hidden_channels = 128
width = 27
height = 24
kernel_sizes, stride_sizes = (3, 3, 3), (1, 1, 1)
train_x = torch.rand((batch_size, num_of_timesteps, width, height, in_channels))
op = CNNOperator(device, in_channels, hidden_channels, kernel_sizes, stride_sizes)
output = op(train_x.to(device))
print(output.size())