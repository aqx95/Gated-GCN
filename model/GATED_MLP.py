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

        self.linear_h = nn.Linear(in_dim, hid_dim)
        self.linear_e = nn.Linear(in_dim_edge, hid_dim)

        self.layers = nn.ModuleList([GatedGCNLayer(hid_dim, out_dim, dropout,
                                                self.graph_norm, self.batch_norm,
                                                self.residual) for _ in range(self.n_layers)])

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(2*out_dim, 200)
        self.bn1 = nn.BatchNorm1d(num_features=200)
        self.output = nn.Linear(200, 1)

    def forward(self, g, node_id, edge_type, norm_n, norm_e, triplets):
        h = self.linear_h(node_id)
        e = self.linear_e(edge_type)

        #convolution
        for conv in self.layers:
            h,e = conv(g, h, e, norm_n, norm_e)

        #FC layer
        s = h[triplets[:,0]]
        o = h[triplets[:,2]]
        conv_map = torch.cat((s,o), dim=1)
        conv_map = self.dropout(conv_map)
        fc1 = nn.ReLU()(self.bn1(self.fc1(conv_map)))
        fc1 = self.dropout(fc1)
        outputs = self.output(fc1)

        return h, outputs

    def get_loss(self, predictions, labels):
        predict_loss = F.binary_cross_entropy_with_logits(predictions, labels)
        return predict_loss
