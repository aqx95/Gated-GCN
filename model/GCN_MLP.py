from dgl.nn import GraphConv
import torch.nn as nn
from model.MLP import MLPPredictor

class GCN_mlp(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, n_hidden_layers):
        super().__init__()
        self.h_embedding = nn.Embedding(in_feat, hid_feat)
        self.mlp = MLPPredictor(hid_feat, out_feat)

        self.layers = nn.ModuleList([GraphConv(hid_feat, hid_feat, allow_zero_in_degree=True)
                                    for _ in range(n_hidden_layers)])

    def forward(self, g, triplets):
        h = self.h_embedding(g.ndata['node_feat'])
        for conv in self.layers:
            h = conv(g, h)

        score = self.mlp(h, triplets)
        return score

    def get_loss(self, predictions, labels):
        predict_loss = nn.CrossEntropyLoss()(predictions, labels)
        return predict_loss
