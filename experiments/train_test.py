import sys
sys.path.insert(1,"../Gated-GCN")

from utilities.utils import load_config
from ogb_data import *
from dgl_data import dgl_data

"""
 Validate if train_data contains test_data
"""
config = load_config('config.yaml')

if config['dataset']['data_name'] == 'fb15k-237':
    train_data, valid_data, test_data, num_nodes, num_rels = dgl_data('fb15k-237')
if config['dataset']['data_name'] == 'biokg':
    train_data, valid_data, test_data, num_nodes, num_rels = prepare_ogb("ogbl-biokg")
if config['dataset']['data_name'] == 'wikikg':
    train_data, valid_data, test_data, num_nodes, num_rels = prepare_ogb("ogbl-wikikg")
if config['dataset']['data_name'] == 'wn18':
    train_data, valid_data, test_data, num_nodes, num_rels = dgl_data('wn18')

for i in test_data:
    if i in train_data:
        print(i)
