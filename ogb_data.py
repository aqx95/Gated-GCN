import ogb
from ogb.linkproppred import DglLinkPropPredDataset

dataset = DglLinkPropPredDataset(name = "ogbl-ppa")

split_edge = dataset.get_edge_split()
train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
graph = dataset[0] # dgl graph object containing only training edges
print(dataset)
