# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class SpatialGCNOperator(nn.Module):
    def __init__(self, DEVICE, in_dims, hidden_dims, adj_mat):
        super(SpatialGCNOperator, self).__init__()
        self.adj_mat = adj_mat
        self.adj_mat_hat(self.adj_mat)
        self.weight = nn.Linear(in_dims, hidden_dims)
        self.adj_mat.to(DEVICE)
        self.to(DEVICE)
    def adj_mat_hat(self, adj_mat):
        adj_mat_sum = torch.sum(adj_mat, dim=1)
        self.adj_mat = self.adj_mat / adj_mat_sum.reshape(-1, 1)
    def forward(self, x):
        bs, n_nodes, in_dims = x.size()
        adj_mat = self.adj_mat.reshape(1, n_nodes, n_nodes)
        adj_mat = adj_mat.repeat(bs, 1, 1)
        out = torch.bmm(adj_mat, x)
        out = self.weight(out)
        return torch.relu(out)
bs = 64
n_nodes = 32
in_dims = 4
hidden_dims = 128
train_x = torch.rand((bs, n_nodes, in_dims))
adj_mat = torch.randint(0, 2, size=(n_nodes, n_nodes), dtype=torch.float)
I = torch.eye(n_nodes)
adj_mat = torch.where(I == 0, adj_mat, I)
op = SpatialGCNOperator(device, in_dims, hidden_dims, adj_mat)
output = op(train_x.to(device))
print(output.size())