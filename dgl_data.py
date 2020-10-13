import dgl
import torch

def dgl_data(name):
  if name == 'wn18':
    data = dgl.data.WN18Dataset(reverse=False)
    graph = data[0]

    num_nodes = graph.number_of_nodes()
    num_rels = len(torch.unique(graph.edata['etype']))

    #extract mask
    train_mask = graph.edata['train_mask']
    valid_mask = graph.edata['val_mask']
    test_mask = graph.edata['test_mask']
    #index
    train_set = torch.arange(graph.number_of_edges())[train_mask]
    valid_set = torch.arange(graph.number_of_edges())[valid_mask]
    test_set = torch.arange(graph.number_of_edges())[test_mask]

    train_data = get_triplets(graph, train_set).numpy()
    valid_data = get_triplets(graph, valid_set)
    test_data = get_triplets(graph, test_set)

    return train_data, valid_data, test_data, num_nodes, num_rels

def get_triplets(data, mask):
    head = data.edges()[0][mask]
    tail = data.edges()[1][mask]
    rel = data.edata['etype'][mask]
    stacked = torch.stack((head, rel, tail), dim=0)
    return torch.transpose(stacked,0,1)
