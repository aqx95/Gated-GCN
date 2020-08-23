import numpy as np
import dgl

# Retreive mapping for entities/relationd dict
def read_dict(filename):
    d = {}
    with open(filename,'r') as file:
        for line in file:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

# Map entities/relation to integers
def read_trip(filename):
    with open(filename, 'r') as file:
        for line in file:
            processed_line = line.strip().split('\t')
            yield processed_line

# Generate triplets
def read_trip_lst(filename, entity_dict, relation_dict):
    lst = []
    for triplet in read_trip(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        lst.append([s, r, o])
    return lst


### Graph utilities

def neg_sampling(pos_sample, num_entity, neg_rate):
    num_edges = len(pos_sample)
    num_samples = num_edges * neg_rate
    neg_samples = np.tile(pos_sample,(neg_rate, 1)) #repeating array A, n number of times
    labels = np.zeros(num_edges * (neg_rate+1), dtype=np.float32)
    labels[:num_edges] = 1
    values = np.random.randint(num_entity, size = num_samples)
    choices = np.random.uniform(size = num_samples)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj,0] = values[subj]
    neg_samples[obj, 2] = values[obj]

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


def build_graph(num_nodes, num_rels, triplets):
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    g.add_edges(src, dst)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel.astype('int64')

def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm
