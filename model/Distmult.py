import torch
import torch.nn as nn


class Distmult(nn.Module):
    def __init__(self, num_ent, num_rel, dim=100):
        self.num_entity = num_ent
        self.num_relation = num_rel
        self.dim = dim

        #Embeddings
        self.entity_embeddings = nn.Embedding(self.num_entity, self.dim)
        self.relation_embeddings = nn.Embedding(self.num_relation, self.dim)

        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)

        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, triplets):
        h = self.entity_embeddings(triplets[:0])
        r = self.relation_embeddings(triplets[:1])
        o = self.entity_embeddings(triplets[:2])

        score = torch.sum(h*r*o, dim=1)
        return score


    def get_loss(self, pred, labels):
        return self.loss(pred, labels)
