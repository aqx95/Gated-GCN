import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MLP import MLPPredictor
from convnet.relg import RELGLayer

class RELG(nn.Module):
    def __init__(self, in_dim, in_dim_edge, hid_dim, out_dim, n_hidden_layers,
                dropout, graph_norm, batch_norm, residual):
        super().__init__()
        self.in_dim = in_dim
        self.in_dim_edge = in_dim_edge
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_hidden_layers
        self.dropout = dropout
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual

        self.h_embedding = nn.Embedding(in_dim, hid_dim)
        self.e_embedding = nn.Embedding(in_dim_edge, hid_dim)

        self.layers = nn.ModuleList([RELGLayer(self.hid_dim, self.hid_dim, self.in_dim_edge,
                                        self.in_dim_edge, self.dropout, self.graph_norm, self.batch_norm,
                                        self.residual) for _ in range(self.n_layers)])

        self.mlp = MLPPredictor(hid_dim, out_dim)


    def forward(self, g, triplets):
        h = self.h_embedding(g.ndata['node_feat'])
        e = self.e_embedding(g.edata['edge_feat'])
        etype = g.edata['edge_feat']

        #convolution
        for conv in self.layers:
            h,e = conv(g, h, e, etype)

        score = self.mlp(h, triplets)
        return score


    def get_loss(self, predictions, labels):
        predict_loss = nn.CrossEntropyLoss()(predictions, labels)
        return predict_loss
