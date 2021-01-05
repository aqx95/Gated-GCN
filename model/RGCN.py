import dgl
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn import RelGraphConv
from model.Bilinear_Pred import BiLinearPredictor

class RGCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, n_layers, num_rel, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layers = n_layers
        self.num_rel = num_rel
        self.dropout = dropout

        self.h_embedding = nn.Embedding(in_dim, hid_dim)
        self.layers = nn.ModuleList([RelGraphConv(
                                    hid_dim, hid_dim, num_rel, regularizer='basis',
                                    num_bases=None,low_mem=True, dropout=self.dropout)
                                    for _ in range(n_layers)])

        self.bilin_score = BiLinearPredictor(num_rel, hid_dim)


    def comp_edge_norm(self, g):
        g_ = g.to(torch.device('cpu'))
        #compute node norm
        in_deg = g_.in_degrees(range(g_.number_of_nodes())).float().numpy()
        node_norm = 1.0 / in_deg
        node_norm[np.isinf(node_norm)] = 0
        node_norm = torch.LongTensor(node_norm)
        #compute edge norm
        g_.ndata['norm'] = node_norm.view(-1,1)
        g_.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
        return g_.edata['norm']


    def forward(self, g, node_id, edge_type):
        h = self.h_embedding(node_id)
        e = edge_type

        edge_norm = self.comp_edge_norm(g)
        if g.device.type == 'cuda':
          edge_norm = edge_norm.cuda()

        for conv in self.layers:
            h = conv(g, h, e, edge_norm)

        return h


    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.distmult.pow(2))

    def get_loss(self, embed, triplets, labels):
        #distmult
        score = self.bilin_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        return predict_loss
