import torch
import numpy as np
from utilities.utils import *
from torch.utils.data import Dataset

class DGLData(Dataset):
    def __init__(self, triplets, num_nodes, num_rel):
        self.triplets = triplets
        self.num_nodes = num_nodes
        self.num_rel = num_rel

    def prep_train_graph(self, sample_size, split_size, neg_rate, sampler='uniform', hot=False, split=True):
        if sampler == 'uniform':
            edges = uniform_sampling(self.triplets, sample_size)
        elif sampler == 'neighbor':
            adj_list, degrees = get_adj_and_degrees(self.num_nodes, self.triplets)
            edges = neighbor_sampling(adj_list, degrees, self.triplets, sample_size)

        # Relabeling of nodes ID
        edges = self.triplets[edges]
        src, rel, dst = edges.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        # Negative sampling
        samples, labels = neg_sampling(relabeled_edges, len(uniq_v), neg_rate)

        if split:
            split_size = int(sample_size * split_size)
            graph_split_ids = np.random.choice(np.arange(sample_size),
                                               size=split_size, replace=False)
            src = src[graph_split_ids]
            dst = dst[graph_split_ids]
            rel = rel[graph_split_ids]

        uniq_v = torch.from_numpy(uniq_v)
        rel = torch.from_numpy(rel)
        g = build_graph(len(uniq_v), (src, rel, dst))
        if hot:
            g = get_onehot_feat(g, self.num_nodes, self.num_rel, uniq_v, rel)
        else:
            g = get_embed_feat(g, self.num_nodes, self.num_rel, uniq_v, rel)

        return g, relabeled_edges, samples, labels

    def prep_test_graph(self, hot=False):
        src, rel, dst = self.triplets.transpose()
        g = build_graph(self.num_nodes, (src,rel,dst))
        #Set node and edge features
        node_id = torch.arange(0, self.num_nodes, dtype=torch.long)
        rel = torch.from_numpy(rel)
        if hot:
            g = get_onehot_feat(g, self.num_nodes, self.num_rel, node_id, rel)
        else:
            g = get_embed_feat(g, self.num_nodes, self.num_rel, node_id, rel)


        return g

if __name__ == '__main__':
    DGLData
