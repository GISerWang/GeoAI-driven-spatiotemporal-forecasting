# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SAttenOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims):
        super(SAttenOperator, self).__init__()
        self.key_layer = nn.Linear(in_dims, hidden_dims)
        self.query_layer = nn.Linear(in_dims, hidden_dims)
        self.value_layer = nn.Linear(in_dims, hidden_dims)
        self.to(DEVICE)
    def forward(self, x):
        # [bs, n_nodes, hidden_dims]
        key = self.key_layer(x)
        # [bs, n_nodes, hidden_dims]
        query = self.query_layer(x)
        # [bs, n_nodes, n_nodes]
        attention_scores = torch.matmul(query, key.permute(0, 2, 1))
        # [bs, n_nodes, n_nodes]
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # [bs, n_nodes, hidden_dims]
        value = self.value_layer(x)
        # [bs, n_nodes, hidden_dims]
        out = torch.matmul(attention_probs, value)
        return torch.relu(out)
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
        out = torch.sigmoid(out)
        # 这里是因果卷积的第二个关键步骤，截取因果卷积应当关注的区域
        return out[:, :, :-self.ps].contiguous()
class ParallelBlock(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, ks):
        super(ParallelBlock, self).__init__()
        self.hidden_dims = hidden_dims
        self.spatial_op = SAttenOperator(DEVICE, in_dims, hidden_dims)
        self.temporal_op = CausalCNOperator(DEVICE, in_dims, hidden_dims, ks)
        self.out_layer = nn.Linear(hidden_dims * 2, hidden_dims)
    def forward(self, x):
        bs, n_timesteps, n_nodes, in_dims = x.size()
        x_s = self.spatial_op(x.reshape(bs * n_timesteps, n_nodes, in_dims))
        x_s = x_s.reshape(bs, n_timesteps, n_nodes, hidden_dims)
        x = x.permute(0, 2, 1, 3)
        x_t = self.temporal_op(x.reshape(bs * n_nodes, n_timesteps, in_dims))
        x_t = x_t.reshape(bs, n_nodes, hidden_dims, n_timesteps)
        x_t = x_t.permute(0, 3, 1, 2)
        out = torch.cat([x_s, x_t], dim=3)
        out = self.out_layer(out)
        return torch.sigmoid(out)
bs = 64
n_timesteps = 10
n_nodes = 32
in_dims = 3
hidden_dims = 128
ks = 3
train_x = torch.rand((bs, n_timesteps, n_nodes, in_dims))
block = ParallelBlock(device, in_dims, hidden_dims, ks)
output = block(train_x.to(device))
print(output.size())