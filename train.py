import os
from data import LinkDataset
import utilities as utils
import numpy as np
import torch
from graphdata import DGLData
from model.GATEDGCN import GatedGCN
import time
import utilities.metrics as eval

class args:
    dropout = 0.4
    n_hidden = 500
    gpu = 0
    lr = 0.01
    n_bases = 100
    n_layers = 2
    n_epochs = 500
    dataset='FB15k-237'
    eval_batch_size = 500
    eval_protocol = 'filtered'
    regularization = 0.01
    grad_norm = 1.0
    graph_batch_size = 30000
    graph_split_size = 0.5
    negative_sample = 5
    eval_every= 250
    edge_sampler = 'uniform'

data = LinkDataset('FB15k-237')
data.load_data()

train_data = data.train
valid_data = data.valid
test_data = data.test
num_nodes = data.num_nodes
num_rels = data.num_rels

# validation and testing triplets
valid_data = torch.LongTensor(valid_data)
test_data = torch.LongTensor(test_data)

 # check cuda
use_cuda =  torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(args.gpu)


#prepare test graph
test_dgl = DGLData(train_data, num_nodes, num_rels)
test_graph, test_rel = test_dgl.prepare_test()
test_deg = test_graph.in_degrees(
            range(test_graph.number_of_nodes())).float().view(-1,1)
test_node_id = torch.arange(0, num_nodes, dtype=torch.long)
test_rel = torch.from_numpy(test_rel)
#test_norm = utils.node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))



train_dgl = DGLData(train_data, num_nodes, num_rels)


# model =  LinkPredict(num_nodes,
#                     args.n_hidden,
#                     num_rels,
#                     num_bases=args.n_bases,
#                     num_hidden_layers=args.n_layers,
#                     dropout=args.dropout,
#                     use_cuda=use_cuda,
#                     reg_param=args.regularization)

model = GatedGCN(num_nodes,
                in_dim_edge=num_rels,
                hid_dim=args.n_hidden,
                out_dim=args.n_hidden,
                n_hidden_layers=args.n_layers,
                dropout=0.2,
                graph_norm=True,
                batch_norm=True,
                residual=True)

# optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=3)

forward_time = []
backward_time = []

# training loop
print("start training...")


best_mrr = 0
for epoch in range(args.n_epochs):
    model.to(args.gpu)
    model.train()
    epoch += 1
    g, node_id, edge_type, data, labels =train_dgl.prepare_train(30000,0.5,args.negative_sample)

    node_id = torch.from_numpy(node_id)
    edge_type = torch.from_numpy(edge_type)
    #edge_norm = utils.node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
    data, labels = torch.from_numpy(data), torch.from_numpy(labels)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
    if use_cuda:
        #deg, edge_norm =  edge_norm.cuda(), deg.cuda()
        data, labels = data.cuda(), labels.cuda()
        g = g.to(args.gpu)

    #Set node and edge features
    node_feat = np.zeros((g.number_of_nodes(), num_nodes))
    node_feat[np.arange(g.number_of_nodes()), node_id] = 1.0
    node_feat = torch.FloatTensor(node_feat)

    edge_feat = np.zeros((g.number_of_edges(), num_rels))
    edge_feat[np.arange(g.number_of_edges()), edge_type] = 1.0
    edge_feat = torch.FloatTensor(edge_feat)

    #norm
    node_norm = 1./((g.number_of_nodes())**0.5)
    edge_norm = 1./((g.number_of_edges())**0.5)

    t0 = time.time()
    embed = model(g, node_id, edge_type)
    loss = model.get_loss(embed, data, labels)
    t1 = time.time()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
    optimizer.step()
    t2 = time.time()

    forward_time.append(t1 - t0)
    backward_time.append(t2 - t1)
    print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
          format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

    optimizer.zero_grad()
    #scheduler.step(loss)
    del g, node_feat, edge_feat, edge_norm, embed

    # validation
    if epoch % args.eval_every == 0:
        #Set node and edge features
        test_node_feat = np.zeros((test_graph.number_of_nodes(), num_nodes))
        test_node_feat[np.arange(test_graph.number_of_nodes()), test_node_id] = 1.0
        test_node_feat = torch.FloatTensor(test_node_feat)

        test_edge_feat = np.zeros((test_graph.number_of_edges(), num_rels))
        test_edge_feat[np.arange(test_graph.number_of_edges()), test_rel] = 1.0
        test_edge_feat = torch.FloatTensor(test_edge_feat)

        #norm
        test_node_norm = 1./((test_graph.number_of_nodes())**0.5)
        test_edge_norm = 1./((test_graph.number_of_edges())**0.5)

        model.cpu()
        model.eval()
        print("start eval")
        with torch.no_grad():
            embed = model(test_graph, test_node_id, test_rel)
            #embed = model(test_graph, test_node_id.cuda(), test_rel.cuda(), test_norm)
            mrr = eval.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
                                 valid_data, test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size,
                                 eval_p=args.eval_protocol)
