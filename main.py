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
from data import LinkDataset
from utilities import utils, metrics
from trainer import Fitter
from graphdata1 import DGLData
from model.GATED_MLP import GatedGCN
from model.GCN import GCN
from model.RGCN import RGCN
from model.RELG_MLP import RELG


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
if config['dataset']['data_name'] == 'fb15k-237':
    train_data, valid_data, test_data, num_nodes, num_rels = prepare_data('FB15k-237')
if config['dataset']['data_name'] == 'biokg':
    train_data, valid_data, test_data, num_nodes, num_rels = prepare_ogb("ogbl-biokg")
if config['dataset']['data_name'] == 'wikikg':
    train_data, valid_data, test_data, num_nodes, num_rels = prepare_ogb("ogbl-wikikg")
if config['dataset']['data_name'] == 'wn18':
    train_data, valid_data, test_data, num_nodes, num_rels = dgl_data('wn18')

# check cuda device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

graph_data = DGLData(train_data, num_nodes, num_rels)
#Prep test
test_graph = graph_data.prep_test_graph()
test_labels = test_data[:,1].unsqueeze(dim=1)
#Prep validation
valid_labels = valid_data[:,1]

# Prepare model
gated = GatedGCN(num_nodes,
                in_dim_edge=num_rels,
                hid_dim=config['model']['n_hidden'],
                out_dim=config['model']['num_class'],
                n_hidden_layers=config['model']['n_layers'],
                dropout=config['model']['dropout'],
                graph_norm=True,
                batch_norm=True,
                residual=True)

rgcn = RGCN(num_nodes, config['model']['n_hidden'],
            config['model']['num_class'],config['model']['n_layers'], num_rels)

gcn = GCN(num_nodes, config['model']['n_hidden'],
            config['model']['num_class'],config['model']['n_layers'])

relg = RELG(num_nodes,
            in_dim_edge=num_rels,
            hid_dim=config['model']['n_hidden'],
            out_dim=config['model']['num_class'],
            n_hidden_layers=config['model']['n_layers'],
            dropout=config['model']['dropout'],
            graph_norm=True,
            batch_norm=True,
            residual=True)

model_zoo = [gated, rgcn, gcn, relg]
epoch_count = range(1, config['train']['n_epochs'] + 1)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
labels = ['GatedGCN', 'RGCN', 'GCN', 'RELG']

## Training
iter = 0
for model in model_zoo:
    print('Training with {}'.format(labels[iter]))
    fitter = Fitter(model, config, device)
    train_loss, val_loss = fitter.fit(graph_data, test_graph, valid_data, test_labels, valid_labels)
    ax1.plot(epoch_count, train_loss, label=labels[iter])
    ax2.plot(epoch_count, val_loss, label=labels[iter])


    ## Inference

    model = model.to('cpu')
    model.eval()
    with torch.no_grad():
        print("Evaluating...")
        pred_test = model(test_graph, test_data)
        print('Results for {}'.format(labels[iter]))
        metrics.get_mrr(pred_test, test_labels, hits=[1,3,10])

    iter += 1
ax1.set_title("Training Loss on "{}.format(config['dataset']['data_name']))
ax1.set_ylabel('Training Loss')
ax1.set_xlabel('Epochs')
ax1.legend()

ax2.set_title("Validation Loss on "{}.format(config['dataset']['data_name']))
ax2.set_ylabel('Validation Loss')
ax2.set_xlabel('Epochs')
ax2.legend()

plt.savefig('loss.png')

# plt.title('Training & Validation Loss on {}'.format(config['dataset']['data_name']))
# plt.xlabel('Epochs')
# plt.ylabel('Training Loss')
# plt.legend()
# plt.savefig('train_loss.png')
