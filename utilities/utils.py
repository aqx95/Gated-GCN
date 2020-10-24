import numpy as np
import dgl
import torch

### Graph utilities

def neg_sampling(pos_sample, num_entity, neg_rate):
    num_edges = len(pos_sample)
    num_samples = num_edges * neg_rate
    neg_samples = np.tile(pos_sample,(neg_rate, 1)) #repeating array A, n number of times
    labels = np.zeros(num_edges * (neg_rate+1), dtype=np.float32 #plus 1 for pos_sample
    labels[:num_edges] = 1
    values = np.random.randint(num_entity, size = num_samples)
    choices = np.random.uniform(size = num_samples)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj,0] = values[subj] #corrupt head
    neg_samples[obj, 2] = values[obj]  #corrupt tail

    return np.concatenate((pos_sample, neg_samples)), labels


def get_adj_and_degrees(num_nodes, triplets):
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees


def uniform_sampling(triplets, sample_size):
    edges = np.arange(len(triplets))
    return np.random.choice(edges, sample_size, replace=False)

def neighbor_sampling(adj_list, degrees, triplets, sample_size):
    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(len(triplets))])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges


def build_graph(num_nodes, triplets):
    src, rel, dst = torch.LongTensor(triplets)
    #g = dgl.graph((src, dst))
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    #print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g


def get_onehot_feat(g, num_nodes, num_edges, node_id, edge_type):
    """ Set node and edge feature using one-hot encoding """
    node_feat = np.zeros((g.number_of_nodes(), num_nodes))
    node_feat[np.arange(g.number_of_nodes()), node_id] = 1.0
    node_feat = torch.FloatTensor(node_feat)

    edge_feat = np.zeros((g.number_of_edges(), num_edges))
    edge_feat[np.arange(g.number_of_edges()), edge_type] = 1.0
    edge_feat = torch.FloatTensor(edge_feat)

    #Add features
    g.ndata['node_feat'] = node_feat
    g.edata['edge_feat'] = edge_feat

    return g

def get_embed_feat(g, num_nodes, num_edges, node_id, edge_type):
    g.ndata['node_feat'] = node_id
    g.edata['edge_feat'] = edge_type

    return g
