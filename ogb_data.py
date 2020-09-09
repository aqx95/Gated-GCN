import ogb
from ogb.linkproppred import DglLinkPropPredDataset
import os, ssl

ssl._create_default_https_context = ssl._create_unverified_context

def prepare_ogb(name):
    dataset = DglLinkPropPredDataset(name)

    split_edge = dataset.get_edge_split()
    train_edge, valid_edge, test_edge = split_edge["train"], split_edge["valid"], split_edge["test"]
    g = dataset[0] # dgl graph object containing only training edges

    num_nodes = g.number_of_nodes()
    num_rel = int(max(torch.unique(g.edata['edge_reltype'])))+1

    edge_type = g.edata['edge_reltype'].squeeze()

    node_feat = torch.arange(0, num_nodes, dtype=torch.long)
    edge_feat = np.zeros((g.number_of_edges(), num_rel))
    edge_feat[np.arange(g.number_of_edges()), edge_type] = 1.0
    edge_feat = torch.FloatTensor(edge_feat)

    #Add features
    g.ndata['node_feat'] = node_feat
    g.edata['edge_feat'] = edge_feat


    return g, train_edge, valid_edge, test_edge, num_nodes, num_rels

if __name__ == "__main__":
    g, train, valid, test, node, rel = prepare_ogb('ogbl-wikikg')
    print(g)
