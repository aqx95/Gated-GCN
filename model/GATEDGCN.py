import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from convnet.gatedgcn import GatedGCNLayer

class GatedGCN(nn.Module):
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

        self.linear_h = nn.Linear(in_dim, hid_dim)
        self.linear_e = nn.Linear(in_dim_edge, hid_dim)

        self.layers = nn.ModuleList([GatedGCNLayer(hid_dim, out_dim, dropout,
                                                    self.graph_norm, self.batch_norm,
                                                    self.residual) for _ in range(self.n_layers)])

        self.w_relation = nn.Parameter(torch.Tensor(self.in_dim_edge, hid_dim))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

        self.h_embedding = nn.Embedding(in_dim, hid_dim)
        self.e_embedding = nn.Embedding(in_dim_edge, hid_dim)


    def forward(self, g, node_id, edge_type):
        h = self.h_embedding(node_id)
        e = self.e_embedding(edge_type)

        #convnets
        for conv in self.layers:
            h, e = conv(g, h, e)

        return h


    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, embed, triplets, labels):
        #distmult
        s = embed[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embed[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss
