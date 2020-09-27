import dgl
from graphdata1 import DGLData
from model.GATED_MLP import GatedGCN_MLP
import time
import os
import torch
import numpy as np
import random
from utilities import utils, metrics
import yaml
from data import LinkDataset
from ogb_data import *
from trainer import Fitter


# Function to load yaml configuration file
def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    random.seed(seed)  #set fixed value for python built-in pseudo-random generator
    np.random.seed(seed) # for numpy pseudo-random generator
    torch.manual_seed(seed) # pytorch (both CPU and CUDA)

def prepare_data(data_name):
    data = LinkDataset(data_name)
    data.load_data()

    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_nodes = data.num_nodes
    num_rels = data.num_rels

    #Convert to Long
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    return train_data, valid_data, test_data, num_nodes, num_rels


config = load_config('config.yaml')
set_seed(config['train']['seed'])

##------ DATA -------
dataset = dgl.data.FB15k237Dataset()
g = dataset[0]

train_mask = g.edata['train_mask']
val_mask = g.edata['val_mask']
test_mask = g.edata['test_mask']

train_set = torch.arange(graph.number_of_edges())[train_mask]
valid_set = torch.arange(graph.number_of_edges())[val_mask]

e_type = g.edata['etype']

# build train_g
train_edges = train_set
train_g = g.edge_subgraph(train_edges, preserve_nodes=True)
train_g.edata['e_type'] = e_type[train_edges];

# build val_g
val_edges = valid_set
val_edges = torch.cat([train_edges, val_edges])
val_g = g.edge_subgraph(val_edges, preserve_nodes=True)
val_g.edata['e_type'] = e_type[val_edges]


sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

dataloader = dgl.dataloading.EdgeDataLoader(
    g, train_edges, sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=4)

model = Model(in_features, hidden_features, out_features, num_classes)
model = model.cuda()
opt = torch.optim.Adam(model.parameters())

for input_nodes, edge_subgraph, blocks in dataloader:
    blocks = [b.to(torch.device('cuda')) for b in blocks]
    edge_subgraph = edge_subgraph.to(torch.device('cuda'))
    input_features = blocks[0].srcdata['features']
    edge_labels = edge_subgraph.edata['labels']
    edge_predictions = model(edge_subgraph, blocks, input_features)
    loss = compute_loss(edge_labels, edge_predictions)
    opt.zero_grad()
    loss.backward()
    opt.step()
