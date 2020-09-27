import torch.nn as nn
import dgl
import torch

class MLPPredictor(nn.Module):
    def __init__(self, in_feat, num_class):
        super().__init__()
        self.W = nn.Linear(in_feat * 2, num_class)

    def apply_edges(self, edges):
        h_src = edges.src['x']
        h_dst = edges.dst['x']
        score = self.W(torch.cat([h_src, h_dst], 1))
        return {'score': score}


    def forward(self, edge_subgraph, x):
        with edge_subgraph.local_scope():
            edge_subgraph.ndata['x'] = x
            edge_subgraph.apply_edges(self.apply_edges)
            return edge_subgraph.edata['score']
