import sys
sys.path.insert(1,"../Gated-GCN")

import time
import os
import torch
import numpy as np
import random
import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


from ogb_data import *
from dgl_data import dgl_data
from utilities import utils, metrics
from trainer import Fitter
from graphdata1 import DGLData
from model.GATED_MLP import GatedGCN_mlp
from model.GCN_MLP import GCN_mlp
from model.RGCN_MLP import RGCN_mlp
from model.RELG_MLP import RELG_mlp



def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    random.seed(seed)  #set fixed value for python built-in pseudo-random generator
    np.random.seed(seed) # for numpy pseudo-random generator
    torch.manual_seed(seed) # pytorch (both CPU and CUDA)


config = utils.load_config('config.yaml')
set_seed(config['train']['seed'])
if config['dataset']['data_name'] == 'fb15k-237':
    train_data, valid_data, test_data, num_nodes, num_rels = dgl_data('fb15k-237')
if config['dataset']['data_name'] == 'biokg':
    train_data, valid_data, test_data, num_nodes, num_rels = prepare_ogb("ogbl-biokg")
if config['dataset']['data_name'] == 'wikikg':
    train_data, valid_data, test_data, num_nodes, num_rels = prepare_ogb("ogbl-wikikg")
if config['dataset']['data_name'] == 'wn18':
    train_data, valid_data, test_data, num_nodes, num_rels = dgl_data('wn18')

print('Training with {}'.format(config['dataset']['data_name']))

# check cuda device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

graph_data = DGLData(train_data, num_nodes, num_rels)
#Prep test
test_graph = graph_data.prep_test_graph()
test_labels = test_data[:,1].unsqueeze(dim=1)
#Prep validation
valid_labels = valid_data[:,1]

# Prepare model
relg1 = RELG_mlp(num_nodes,
            in_dim_edge=num_rels,
            hid_dim=config['model']['n_hidden'],
            out_dim=config['model']['num_class'],
            n_hidden_layers=1,
            dropout=config['model']['dropout'],
            graph_norm=True,
            batch_norm=True,
            residual=True)

relg2 = RELG_mlp(num_nodes,
            in_dim_edge=num_rels,
            hid_dim=config['model']['n_hidden'],
            out_dim=config['model']['num_class'],
            n_hidden_layers=2,
            dropout=config['model']['dropout'],
            graph_norm=True,
            batch_norm=True,
            residual=True)

relg3 = RELG_mlp(num_nodes,
            in_dim_edge=num_rels,
            hid_dim=config['model']['n_hidden'],
            out_dim=config['model']['num_class'],
            n_hidden_layers=3,
            dropout=config['model']['dropout'],
            graph_norm=True,
            batch_norm=True,
            residual=True)

relg5 = RELG_mlp(num_nodes,
            in_dim_edge=num_rels,
            hid_dim=config['model']['n_hidden'],
            out_dim=config['model']['num_class'],
            n_hidden_layers=5,
            dropout=config['model']['dropout'],
            graph_norm=True,
            batch_norm=True,
            residual=True)

model_zoo = [relg1, relg2, relg3, relg5]
epoch_count = range(1, config['train']['n_epochs'] + 1)

labels = ['RELG1', 'RELG2', 'RELG3', 'RELG5']

## Training
iter = 0
for model in model_zoo:
    print('Training with {}'.format(labels[iter]))
    print('Number of parameters: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    fitter = Fitter(model, config, device)
    train_loss, val_loss = fitter.fit(graph_data, test_graph, valid_data, test_labels, valid_labels)
    plt.plot(epoch_count, train_loss, label=labels[iter])


    ## Inference
    model = model.to('cpu')
    model.eval()
    with torch.no_grad():
        print("Evaluating...")
        pred_test = model(test_graph, test_data)
        print('Results for {}'.format(labels[iter]))
        metrics.get_mrr(pred_test, test_labels, hits=[1,3,10])

    iter += 1

plt.savefig('loss.png', bbox_inches = "tight")

plt.title('Training & Validation Loss on {}'.format(config['dataset']['data_name']))
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.legend()
plt.savefig('train_loss.png')
