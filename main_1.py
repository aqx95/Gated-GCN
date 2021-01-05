import os
import torch
import random
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

from ogb_data import prepare_ogb
from dgl_data import dgl_data
from utilities import utils, metrics
from trainer1 import Fitter
from graphdata import DGLData

from model.GATEDGCN import GatedGCN
from model.RGCN import RGCN
from model.RELG import RELG


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    config_path = 'config_' + args.dataset + '.yaml'
    config = utils.load_config(config_path)

    set_seed(2020)
    if config['dataset']['data_name'] == 'fb15k-237':
        train_data, valid_data, test_data, num_nodes, num_rels = dgl_data('fb15k-237')
    if config['dataset']['data_name'] == 'biokg':
        train_data, valid_data, test_data, num_nodes, num_rels = prepare_ogb("ogbl-biokg")
    if config['dataset']['data_name'] == 'wikikg':
        train_data, valid_data, test_data, num_nodes, num_rels = prepare_ogb("ogbl-wikikg")
    if config['dataset']['data_name'] == 'wn18':
        train_data, valid_data, test_data, num_nodes, num_rels = dgl_data('wn18')

    # Use available device (CPU/GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Preprocess Data
    graph_data = DGLData(train_data, num_nodes, num_rels)
    # Prepare test and validation graph
    test_graph, test_node_id, test_rel = graph_data.prep_test_graph()
    test_data = torch.LongTensor(test_data)
    test_labels = test_data[:,1].unsqueeze(dim=1)
    valid_labels = valid_data[:,1]

    # Prepare Models
    gated = GatedGCN(num_nodes,
                    in_dim_edge=num_rels,
                    hid_dim=config['model']['n_hidden'],
                    out_dim=config['model']['n_hidden'],
                    n_hidden_layers=config['model']['n_layers'],
                    dropout=config['model']['dropout'],
                    graph_norm=True,
                    batch_norm=True,
                    residual=True)

    rgcn = RGCN(num_nodes,
                config['model']['n_hidden'],
                config['model']['n_hidden'],
                config['model']['n_layers'],
                num_rels,
                config['model']['dropout'])

    relg = RELG(num_nodes,
                in_dim_edge=num_rels,
                hid_dim=config['model']['n_hidden'],
                out_dim=config['model']['n_hidden'],
                n_hidden_layers=config['model']['n_layers'],
                dropout=config['model']['dropout'],
                graph_norm=True,
                batch_norm=True,
                residual=True)

    model_zoo = [relg]
    epoch_count = range(1, config['train']['n_epochs'] + 1)

    labels = ['RELG']

    # Training
    iter = 0
    for model in model_zoo:
        print('Training with {}'.format(labels[iter]))
        fitter = Fitter(model, config, device)
        train_loss = fitter.fit(graph_data, test_graph, valid_data, valid_labels)
        plt.plot(epoch_count, train_loss, label=labels[iter])

        # Inference
        # if args.dataset not in ["fb15k", "wn18"]:
        #     model = model.to('cpu')
        # if args.dataset in ["fb15k", "wn18"]:
        #     train_data = torch.from_numpy(train_data)
        #     train_data, test_graph = train_data.to(device), test_graph.to(device)
        #     test_node_id, test_rel = test_node_id.to(device), test_rel.to(device)
        model.cpu()
        model.eval()
        print('Start evaluating...')
        with torch.no_grad():
            embed = model(test_graph, test_node_id, test_rel)
            mrr = metrics.calc_mrr(embed.detach(), model.bilin_score.W.detach(), torch.LongTensor(train_data),
                                 valid_data, test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size,
                                 eval_p=args.eval_protocol)

        iter += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RELG')
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=500,
            help="batch size when evaluating")
    parser.add_argument("--eval-protocol", type=str, default="filtered",
            help="type of evaluation protocol: 'raw' or 'filtered' mrr")

    args = parser.parse_args()
    print(args)
    main(args)
