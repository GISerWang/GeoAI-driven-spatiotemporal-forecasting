# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class GATOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, adj_mat, n_nodes):
        super(GATOperator, self).__init__()
        self.key_layer = nn.Linear(in_dims, hidden_dims)
        self.query_layer = nn.Linear(in_dims, hidden_dims)
        self.value_layer = nn.Linear(in_dims, hidden_dims)
        self.minimum = torch.full((n_nodes, n_nodes), -1e11)
        self.adj_mat = adj_mat
        self.minimum.to(DEVICE)
        self.adj_mat.to(DEVICE)
        self.to(DEVICE)
    def forward(self, x):
        bs, n_nodes, hidden_dims = x.size()
        # [bs, n_nodes, hidden_dims]
        key = self.key_layer(x)
        # [bs, n_nodes, hidden_dims]
        query = self.query_layer(x)
        # [bs, n_nodes, n_nodes]
        attention_scores = torch.matmul(query, key.permute(0, 2, 1))
        attention_scores = torch.where(self.adj_mat.repeat(bs, 1, 1) == 1,
                                       attention_scores,
                                       self.minimum.repeat(bs, 1, 1))
        # [bs, n_nodes, n_nodes]
        attention_probs = torch.softmax(attention_scores, dim=-1)
        # [bs, n_nodes, hidden_dims]
        value = self.value_layer(x)
        # [bs, n_nodes, hidden_dims]
        out = torch.matmul(attention_probs, value)
        return torch.relu(out)
bs = 64
n_nodes = 32
in_dims = 4
hidden_dims = 128
adj_mat = torch.randint(0, 2, size=(n_nodes, n_nodes), dtype=torch.float)
I = torch.eye(n_nodes)
adj_mat = torch.where(I == 0, adj_mat, I)
train_x = torch.rand((bs, n_nodes, in_dims))
op = GATOperator(device, in_dims, hidden_dims, adj_mat, n_nodes)
output = op(train_x.to(device))
print(output.size())