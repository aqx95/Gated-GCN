import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.MLP import MLPPredictor
from convnet.relg import RELGLayer

class RELG_mlp(nn.Module):
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

        #self.mlp = MLPPredictor(hid_dim, out_dim)
        self.distmult = nn.Parameter(torch.Tensor(in_dim_edge, hid_dim))

    def forward(self, g, triplets):
        h = self.h_embedding(g.ndata['node_feat'])
        e = self.e_embedding(g.edata['edge_feat'])
        etype = g.edata['edge_feat']

        #convolution
        for conv in self.layers:
            h,e = conv(g, h, e, etype)

        #score = self.mlp(h, triplets)
        return h#score


    def calc_score(self, embedding, triplets):
        s = embedding[triplets[:,0]]
        r = self.distmult[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score


    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.distmult.pow(2))


    def get_loss(self, g, embed, triplets, labels):
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + 0.01 * reg_loss

    # def get_loss(self, predictions, labels):
    #     predict_loss = nn.CrossEntropyLoss()(predictions, labels)
    #     return predict_loss
