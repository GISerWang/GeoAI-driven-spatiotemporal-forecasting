# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        attention_probs = F.softmax(attention_scores, dim=-1)
        # [bs, n_nodes, hidden_dims]
        value = self.value_layer(x)
        # [bs, n_nodes, hidden_dims]
        out = torch.matmul(attention_probs, value)
        return torch.relu(out)
bs = 64
n_nodes = 32
in_dims = 4
hidden_dims = 128
train_x = torch.rand((bs, n_nodes, in_dims))
op = SAttenOperator(device, in_dims, hidden_dims)
output = op(train_x.to(device))
print(output.size())