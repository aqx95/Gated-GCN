import torch.nn as nn
import dgl
import torch

class MLPPredictor(nn.Module):
    def __init__(self, in_feat, num_class):
        super().__init__()
        self.W = nn.Linear(in_feat * 2, num_class)

    def forward(self, x, triplets):
        s = x[triplets[:,0]]
        o = x[triplets[:,2]]
        conv_map = torch.cat((s,o), dim=1)
        return self.W(conv_map)
