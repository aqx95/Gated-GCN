from dgl.nn import RelGraphConv
import torch.nn as nn
from model.MLP import MLPPredictor

class RGCN(nn.Module):
    def __init__(self, in_feat, hid_feat, out_feat, n_hidden_layers, num_rel):
        super().__init__()
        self.h_embedding = nn.Embedding(in_feat, hid_feat)
        self.mlp = MLPPredictor(hid_feat, out_feat)

        self.layers = nn.ModuleList([RelGraphConv(
                                    hid_feat, hid_feat, num_rel, low_mem=True, dropout=0.4)
                                    for _ in range(n_hidden_layers)])

    def forward(self, g, triplets):
        h = self.h_embedding(g.ndata['node_feat'])
        e = g.edata['edge_feat']
        for conv in self.layers:
            h = conv(g, h, e)

        score = self.mlp(h, triplets)
        return score

    def get_loss(self, predictions, labels):
        predict_loss = nn.CrossEntropyLoss()(predictions, labels)
        return predict_loss
