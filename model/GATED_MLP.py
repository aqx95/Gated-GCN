import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from convnet.gatedgcn import GatedGCNLayer

class GatedGCN_MLP(nn.Module):
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
        self.entity_embedding = nn.Parameter(torch.zeros(in_dim, hid_dim))
        nn.init.uniform_(tensor=self.entity_embedding)
        self.relation_embedding = nn.Parameter(torch.zeros(in_dim_edge, hid_dim))
        nn.init.uniform_(tensor=self.relation_embedding)
        # self.linear_h = nn.Linear(in_dim, hid_dim)
        # self.linear_e = nn.Linear(in_dim_edge, hid_dim)

        self.layers = nn.ModuleList([GatedGCNLayer(self.hid_dim, self.hid_dim,
                                        self.dropout, self.graph_norm, self.batch_norm,
                                        self.residual) for _ in range(self.n_layers)])

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2*self.hid_dim, 1000)
        self.bn1 = nn.BatchNorm1d(num_features=1000)
        self.output = nn.Linear(1000, self.out_dim)


    def forward(self, g, norm_n, norm_e, triplets):

        # h = self.linear_h(g.ndata['node_feat'])
        # e = self.linear_e(g.edata['edge_feat'])
        h = self.h_embedding(g.ndata['node_feat'])
        e = self.e_embedding(g.edata['edge_feat'])
        # h = torch.index_select(
        #     self.entity_embedding,
        #     dim=0,
        #     index=g.ndata['node_feat'])
        #
        # e = torch.index_select(
        #         self.relation_embedding,
        #         dim=0,
        #         index=g.edata['edge_feat'])

        #convolution
        for conv in self.layers:
            h,e = conv(g, h, e, norm_n, norm_e)

        # #FC layer
        s = h[triplets[:,0]]
        o = h[triplets[:,2]]
        conv_map = torch.cat((s,o), dim=1)
        conv_map = self.dropout(conv_map)
        fc1 = nn.ReLU()(self.bn1(self.fc1(conv_map)))
        fc1 = self.dropout(fc1)
        outputs = self.output(fc1)

        return outputs


    def get_loss(self, predictions, labels):
        predict_loss = nn.CrossEntropyLoss()(predictions, labels)
        return predict_loss
