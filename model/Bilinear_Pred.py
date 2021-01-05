import torch
import torch.nn as nn


class BiLinearPredictor(nn.Module):
    def __init__(self, num_rel, feat_dim):
        super().__init__()
        self.W = nn.Parameter(torch.Tensor(num_rel, feat_dim))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))

    def forward(self, h, triplets):
        hs = h[triplets[:,0]]
        Wr = self.W[triplets[:,1]]
        ho = h[triplets[:,2]]
        score = torch.sum(hs*Wr*ho, dim=1)
        return score
