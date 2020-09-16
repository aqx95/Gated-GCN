import ogb
import torch
import numpy as np
from ogb.linkproppred import DglLinkPropPredDataset
import os, ssl

ssl._create_default_https_context = ssl._create_unverified_context

def prepare_ogb(name):
    dataset = DglLinkPropPredDataset(name)

    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
    g = dataset[0] # dgl graph object containing only training edges

    train_data = sample_data(50000, train_edge, True).numpy()
    valid_data = sample_data(50000, valid_edge)
    test_data = sample_data(50000, test_edge)

    num_nodes = g.number_of_nodes()
    num_rels = len(torch.unique(train_edge['relation']))


    return train_data, valid_data, test_data, num_nodes, num_rels


def sample_data(size, edge, sampling=False):
    if sampling:
        sample = np.random.choice(range(edge['head'].size()[0]), size, replace=False)
        head = edge['head'][sample]
        relation = edge['relation'][sample]
        tail = edge['tail'][sample]

    else:
        head = edge['head']
        relation = edge['relation']
        tail = edge['tail']

    stacked = torch.stack((head, relation, tail), dim=0)
    return torch.transpose(stacked,0,1)
