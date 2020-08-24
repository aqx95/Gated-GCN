import torch
import numpy as np
from utilities.utils import *
from torch.utils.data import Dataset

class DGLData(Dataset):
    def __init__(self, triplets, num_nodes, num_rel):
        self.triplets = triplets
        self.num_nodes = num_nodes
        self.num_rel = num_rel

    def prepare_train(self, sample_size, split_size, neg_rate, sampler='uniform'):
        adj_list, degrees = get_adj_and_degrees(self.num_nodes, self.triplets)

        if sampler == 'uniform':
            edges = uniform_sampling(self.triplets, sample_size)
        elif sampler == 'neighbor':
            edges = neighbor_sampling(adj_list, degrees, self.triplets, sample_size)

        # Relabeling of nodes ID
        edges = self.triplets[edges]
        src, rel, dst = edges.transpose()
        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        src, dst = np.reshape(edges, (2, -1))
        relabeled_edges = np.stack((src, rel, dst)).transpose()

        # build DGL graph
        print("# sampled nodes: {}".format(len(uniq_v)))
        print("# sampled edges: {}".format(len(src)))
        g, rel = build_graph(len(uniq_v), self.num_rel,
                                    (src, rel, dst))

        return g, uniq_v, rel, relabeled_edges

    def prepare_test(self):
        src, rel, dst = self.triplets.transpose()
        return build_graph(self.num_nodes, self.num_rel, (src,rel,dst))




if __name__ == '__main__':
    DGLData
