# import os
# from data import LinkDataset
# import numpy as np
# import torch.nn as nn
# import torch
# from graphdata1 import DGLData
# from model.GATED_MLP import GatedGCN_MLP
# import time
# import random
# from utilities import utils, metrics
# import yaml
#
# # Function to load yaml configuration file
# def load_config(config_name):
#     with open(config_name) as file:
#         config = yaml.safe_load(file)
#
#     return config
#
# def set_seed(seed):
#     os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED env var at fixed value
#     random.seed(seed)  #set fixed value for python built-in pseudo-random generator
#     np.random.seed(seed) # for numpy pseudo-random generator
#     torch.manual_seed(seed) # pytorch (both CPU and CUDA)
#
# def prepare_data(data_name):
#     data = LinkDataset(data_name)
#     data.load_data()
#
#     train_data = data.train
#     valid_data = data.valid
#     test_data = data.test
#     num_nodes = data.num_nodes
#     num_rels = data.num_rels
#
#     #Convert to Long
#     valid_data = torch.LongTensor(valid_data)
#     test_data = torch.LongTensor(test_data)
#
#     return train_data, valid_data, test_data, num_nodes, num_rels
#
#
#
# config = load_config('config.yaml')
# set_seed(config['train']['seed'])
# train_data, valid_data, test_data, num_nodes, num_rels = prepare_data(config['dataset']['data_name'])
#
# # check cuda
# use_cuda =  torch.cuda.is_available()
# if use_cuda:
#     torch.cuda.set_device(config['train']['gpu'])
#
# graph_data = DGLData(train_data, num_nodes, num_rels)
# #Prep test
# test_graph = graph_data.prep_test_graph()
# test_labels = test_data[:,1].unsqueeze(dim=1)
# #Prep validation
# valid_labels = valid_data[:,1]
# # #prepare test graph
# # test_dgl = DGLData(train_data, num_nodes, num_rels)
# # test_graph, test_rel = test_dgl.prepare_test()
# # # test_deg = test_graph.in_degrees(
# # #             range(test_graph.number_of_nodes())).float().view(-1,1)
# # test_node_id = torch.arange(0, num_nodes, dtype=torch.long)
# # test_rel = torch.from_numpy(test_rel)
#
#
#
#
# #validation
# # valid_dgl = DGLData(train_data, num_nodes, num_rels)
# # valid_graph, valid_rel = valid_dgl.prepare_test()
# # # valid_deg = valid_graph.in_degrees(
# # #             range(valid_graph.number_of_nodes())).float().view(-1,1)
# # valid_node_id = torch.arange(0, num_nodes, dtype=torch.long)
# # valid_rel = torch.from_numpy(valid_rel)
# # valid_labels = valid_data[:,1]
#
# model = GatedGCN_MLP(num_nodes,
#                 in_dim_edge=num_rels,
#                 hid_dim=config['model']['n_hidden'],
#                 out_dim=config['model']['num_class'],
#                 n_hidden_layers=config['model']['n_layers'],
#                 dropout=config['model']['dropout'],
#                 graph_norm=True,
#                 batch_norm=True,
#                 residual=True)
#
# if use_cuda:
#     model.cuda()
#
# # optimizer & scheduler
# optimizer = torch.optim.Adam(model.parameters(), lr=config['optimizer']['lr'],
#                              weight_decay=config['optimizer']['regularization'])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=3)
#
# forward_time = []
# backward_time = []
#
# # training loop
# print("start training...")
#
#
# best_mrr = 0
# for epoch in range(config['train']['n_epochs']):
#     model.train()
#     epoch += 1
#     g, data = graph_data.prep_train_graph(config['graph_obj']['batch_size'],
#                                           config['graph_obj']['split_size'],
#                                           config['graph_obj']['neg_sampling'])
#
#     labels = torch.LongTensor(data[:,1])
#     # node_id = torch.from_numpy(node_id)
#     # edge_type = torch.from_numpy(edge_type)
#     # data = torch.from_numpy(data)
#     # deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
#     if use_cuda:
#         g, labels = g.cuda(), labels.cuda()
#
#     #Set node and edge features
#     # node_feat = np.zeros((g.number_of_nodes(), num_nodes))
#     # node_feat[np.arange(g.number_of_nodes()), node_id] = 1.0
#     # node_feat = torch.FloatTensor(node_feat)
#     #
#     # edge_feat = np.zeros((g.number_of_edges(), num_rels))
#     # edge_feat[np.arange(g.number_of_edges()), edge_type] = 1.0
#     # edge_feat = torch.FloatTensor(edge_feat)
#
#     # print(data)
#     # for i in data:
#     #     if(torch.equal(i,test_data[2])):
#     #         print(i)
#
#
#     #norm
#     node_norm = 1./((g.number_of_nodes())**0.5)
#     edge_norm = 1./((g.number_of_edges())**0.5)
#
#     t0 = time.time()
#     pred = model(g, node_norm, edge_norm, data)
#     t_loss = model.get_loss(pred, labels)
#     t1 = time.time()
#     t_loss.backward()
#     torch.nn.utils.clip_grad_norm_(model.parameters(), config['optimizer']['grad_norm'])
#     optimizer.step()
#     t2 = time.time()
#
#
#
#     forward_time.append(t1 - t0)
#     backward_time.append(t2 - t1)
#     print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
#           format(epoch, t_loss.item(), best_mrr, forward_time[-1], backward_time[-1]))
#
#     optimizer.zero_grad()
#     del g, labels, data
#
#     # validation
#     # if epoch % args.eval_every == 0:
#         #Set node and edge features
#
#     #norm
#     valid_node_norm = 1./((test_graph.number_of_nodes())**0.5)
#     valid_edge_norm = 1./((test_graph.number_of_edges())**0.5)
#
#     if use_cuda:
#         valid_data = valid_data.cuda()
#         valid_labels = valid_labels.cuda()
#
#     model.eval()
#     with torch.no_grad():
#         pred = model(test_graph, valid_node_norm, valid_edge_norm, valid_data)
#         loss = model.get_loss(pred, valid_labels)
#         print("Epoch {:04d} | Loss {:.4f} |".format(epoch, loss.item()))
#
#         metrics.get_mrr(pred, valid_labels.unsqueeze(dim=1), hits=[1,3,10])
#
#     scheduler.step(loss)
#
#
# #test time
# #norm
# test_node_norm = 1./((test_graph.number_of_nodes())**0.5)
# test_edge_norm = 1./((test_graph.number_of_edges())**0.5)
#
# if use_cuda:
#     test_data = test_data.cuda()
#     test_labels = test_labels.cuda()
#
# model.eval()
# with torch.no_grad():
#     pred_test = model(test_graph, test_node_norm,test_edge_norm,
#                       torch.LongTensor([[4092, 23, 13397]]))
#
#     soft = nn.Softmax(dim=1)
#     scores = soft(pred_test)
#     _, indices = torch.sort(scores, descending=True)
#     rank = torch.where((indices == test_labels[0]))[1] + 1
#     print(rank.item())
