# model/gnn_model.py

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class GNN(nn.Module):
    def __init__(self, metadata, hidden_channels=64, out_channels=2, data=None):
        super().__init__()

        self.lin_in = nn.ModuleDict()
        for node_type in metadata[0]:  # node types
            input_dim = data[node_type].x.size(1)
            self.lin_in[node_type] = nn.Linear(input_dim, hidden_channels)

        self.conv1 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]  # edge types
        }, aggr='mean')

        self.conv2 = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_channels)
            for edge_type in metadata[1]
        }, aggr='mean')

        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.act = nn.ReLU()

    def forward(self, x_dict, edge_index_dict):
        x_dict = {
            node_type: self.act(self.lin_in[node_type](x))
            for node_type, x in x_dict.items()
        }

        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.act(v) for k, v in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)

        return {'article': self.lin_out(x_dict['article'])}