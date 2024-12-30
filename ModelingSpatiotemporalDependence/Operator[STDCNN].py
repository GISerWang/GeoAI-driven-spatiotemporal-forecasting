# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class STDCNNOperator(nn.Module):
    def __init__(self, DEVICE, in_channels, hidden_channels, kernel_sizes, stride_sizes, dilation_sizes):
        super(STDCNNOperator, self).__init__()
        self.padding_sizes = ((kernel_sizes[0] - 1) * dilation_sizes[0], int((kernel_sizes[1] - 1) * dilation_sizes[1] / 2), int((kernel_sizes[2] - 1) * dilation_sizes[2] / 2))
        self.conv3d = nn.Conv3d(in_channels, hidden_channels, kernel_size=kernel_sizes, padding=self.padding_sizes, stride=stride_sizes, dilation=dilation_sizes)
        self.to(DEVICE)
    def forward(self, x):
        # [batch_size, num_of_timesteps, width, height, in_channels] -> [batch_size, in_channels, num_of_timesteps, width, height]
        x = x.permute(0, 4, 1, 2, 3)
        # [batch_size, hidden_channels, num_of_timesteps + padding_sizes[0], width, height]
        out = self.conv3d(x)
        # [batch_size, hidden_channels, num_of_timesteps, width, height]
        return torch.relu(out[:, :, :-self.padding_sizes[0], :, :])

batch_size = 64
num_of_timesteps = 10
in_channels = 1
hidden_channels = 128
width = 27
height = 24
kernel_sizes, dilation_sizes, stride_sizes = (3, 3, 3), (2, 2, 2), (1, 1, 1)
train_x = torch.rand((batch_size, num_of_timesteps, width, height, in_channels))
op = STDCNNOperator(device, in_channels, hidden_channels, kernel_sizes, stride_sizes, dilation_sizes)
output = op(train_x.to(device))
print(output.size())