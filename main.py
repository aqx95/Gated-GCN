from graphdata1 import DGLData
from model.GATED_MLP import GatedGCN
from model.GCN import GCN
from model.RGCN import RGCN
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
train_data, valid_data, test_data, num_nodes, num_rels = prepare_data(config['dataset']['data_name'])
#train_data, valid_data, test_data, num_nodes, num_rels = prepare_ogb("ogbl-biokg")


# check cuda device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

graph_data = DGLData(train_data, num_nodes, num_rels)
#Prep test
test_graph = graph_data.prep_test_graph()
#test_graph = test_graph.to(device)
test_labels = test_data[:,1].unsqueeze(dim=1)
#Prep validation
valid_labels = valid_data[:,1]

# Prepare model
# model = GatedGCN(num_nodes,
#                 in_dim_edge=num_rels,
#                 hid_dim=config['model']['n_hidden'],
#                 out_dim=config['model']['num_class'],
#                 n_hidden_layers=config['model']['n_layers'],
#                 dropout=config['model']['dropout'],
#                 graph_norm=True,
#                 batch_norm=True,
#                 residual=True)
model = RGCN(num_nodes, config['model']['n_hidden'],
            config['model']['num_class'],config['model']['n_layers'], num_rels)

# model = LinkPredict(num_nodes,
#                     config['model']['n_hidden'],
#                     num_rels,
#                     num_bases=100,
#                     num_hidden_layers=config['model']['n_layers'],
#                     dropout=config['model']['dropout'],
#                     use_cuda=0,
#                     reg_param=0.01)

## Training
fitter = Fitter(model, config, device)
fitter.fit(graph_data, test_graph, valid_data, test_labels, valid_labels)


## Inference
print("Evaluation with test set")
# test_node_norm = 1./((test_graph.number_of_nodes())**0.5)
# test_edge_norm = 1./((test_graph.number_of_edges())**0.5)

#test_data, test_labels = test_data.to(device), test_labels.to(device)
model = model.to('cpu')
model.eval()
with torch.no_grad():
    print("Evaluating...")
    pred_test = model(test_graph, test_data)
    metrics.get_mrr(pred_test, test_labels, hits=[1,3,10])
