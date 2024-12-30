# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CausalCNOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, ks):
        super(CausalCNOperator, self).__init__()
        # 这里是因果卷积的第一个关键步骤，比正常1DCNN填充更多的空白区域
        # padding size(ps)通过kernel size(ks)计算
        self.ps = ks - 1
        self.causal_conv = nn.Conv1d(in_dims, hidden_dims, ks, padding=self.ps)
        self.to(DEVICE)
    def forward(self, x):
        # [bs, n_timesteps, in_dims] -> [bs, in_dims, n_timesteps]
        x = x.permute(0, 2, 1)
        # 利用pytorch封装的函数实现causal_conv算子
        # [bs, hidden_dims, n_timesteps + ps]
        out = self.causal_conv(x)
        out = F.relu(out)
        # 这里是因果卷积的第二个关键步骤，截取因果卷积应当关注的区域
        return out[:, :, :-self.ps].contiguous()
bs = 64
n_timesteps = 10
in_dims = 3
hidden_dims = 128
# 卷积核的大小为3
ks = 3
# 构建一个模拟的数据，数据形状为(n_timesteps, in_dims)，共有bs个
train_x = torch.rand((bs, n_timesteps, in_dims))
op = CausalCNOperator(device, in_dims, hidden_dims, ks)
output = op(train_x.to(device))
print(output.size())
