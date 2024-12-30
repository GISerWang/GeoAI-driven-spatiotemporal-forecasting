# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CausalDCNOperator(nn.Module):
    def __init__(self, DEVICE, in_channels, hidden_channels, kernel_size, stride_size, dilation_size):
        super(CausalDCNOperator, self).__init__()
        self.padding_size = (kernel_size - 1) * dilation_size
        self.causal_conv = nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=self.padding_size, stride=stride_size, dilation=dilation_size)
        self.to(DEVICE)
    def forward(self, x):
        # [batch_size, num_of_timesteps, in_channels] -> [batch_size, in_channels, num_of_timesteps]
        x = x.permute(0, 2, 1)
        # [batch_size, hidden_channels, num_of_timesteps + padding_size]
        out = self.causal_conv(x)
        out = F.relu(out)
        return out[:, :, :-self.padding_size].contiguous()
batch_size = 64
num_of_timesteps = 10
in_channels = 1
hidden_channels = 128
kernel_size, dilation_size, stride_size = 3, 4, 1
train_x = torch.rand((batch_size, num_of_timesteps, in_channels))
op = CausalDCNOperator(device, in_channels, hidden_channels, kernel_size, stride_size, dilation_size)
output = op(train_x.to(device))
print(output.size())