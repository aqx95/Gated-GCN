import os
from data import LinkDataset
import numpy as np
import torch
from graphdata1 import DGLData
from model.GATED_MLP import GatedGCN_MLP
import time
import random
from utilities import utils, metrics

class args:
    dropout = 0.4
    n_hidden = 500
    gpu = 0
    lr = 0.01
    n_bases = 100
    n_layers = 1
    n_epochs = 200
    dataset='FB15k-237'
    eval_batch_size = 500
    eval_protocol = 'filtered'
    regularization = 0.01
    grad_norm = 1.0
    graph_batch_size = 30000
    graph_split_size = 0.5
    negative_sample = 5
    eval_every= 100
    edge_sampler = 'uniform'

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    random.seed(seed)  #set fixed value for python built-in pseudo-random generator
    np.random.seed(seed) # for numpy pseudo-random generator
    torch.manual_seed(seed) # pytorch (both CPU and CUDA)

set_seed(2020)

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



train_dgl = DGLData(train_data, num_nodes, num_rels)

#validation
valid_dgl = DGLData(train_data, num_nodes, num_rels)
valid_graph, valid_rel = valid_dgl.prepare_test()
valid_deg = valid_graph.in_degrees(
            range(valid_graph.number_of_nodes())).float().view(-1,1)
valid_node_id = torch.arange(0, num_nodes, dtype=torch.long)
valid_rel = torch.from_numpy(valid_rel)
valid_labels = valid_data[:,1]

model = GatedGCN_MLP(num_nodes,
                in_dim_edge=num_rels,
                hid_dim=args.n_hidden,
                out_dim=args.n_hidden,
                n_hidden_layers=args.n_layers,
                dropout=0.2,
                graph_norm=True,
                batch_norm=True,
                residual=True)

if use_cuda:
    model.cuda()

# optimizer & scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,patience=3)

forward_time = []
backward_time = []

# training loop
print("start training...")


best_mrr = 0
for epoch in range(args.n_epochs):
    model.train()
    epoch += 1
    g, node_id, edge_type, data = train_dgl.prepare_train(30000,0.5,10)

    labels = torch.LongTensor(data[:,1])
    node_id = torch.from_numpy(node_id)
    edge_type = torch.from_numpy(edge_type)
    data = torch.from_numpy(data)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
    if use_cuda:
        deg, data, labels = deg.cuda(), data.cuda(), labels.cuda()
        g = g.to(args.gpu)

    #Set node and edge features
    node_feat = np.zeros((g.number_of_nodes(), num_nodes))
    node_feat[np.arange(g.number_of_nodes()), node_id] = 1.0
    node_feat = torch.FloatTensor(node_feat)

    edge_feat = np.zeros((g.number_of_edges(), num_rels))
    edge_feat[np.arange(g.number_of_edges()), edge_type] = 1.0
    edge_feat = torch.FloatTensor(edge_feat)

    if use_cuda:
        node_feat = node_feat.cuda()
        edge_feat = edge_feat.cuda()

    #norm
    node_norm = 1./((g.number_of_nodes())**0.5)
    edge_norm = 1./((g.number_of_edges())**0.5)

    t0 = time.time()
    pred = model(g, node_feat, edge_feat, node_norm, edge_norm, data)
    loss = model.get_loss(pred, labels)
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
    scheduler.step(loss)
    del g, node_feat, edge_feat

    # validation
    # if epoch % args.eval_every == 0:
        #Set node and edge features
    valid_node_feat = np.zeros((valid_graph.number_of_nodes(), num_nodes))
    valid_node_feat[np.arange(valid_graph.number_of_nodes()), valid_node_id] = 1.0
    valid_node_feat = torch.FloatTensor(valid_node_feat)

    valid_edge_feat = np.zeros((valid_graph.number_of_edges(), num_rels))
    valid_edge_feat[np.arange(valid_graph.number_of_edges()), valid_rel] = 1.0
    valid_edge_feat = torch.FloatTensor(valid_edge_feat)

    #norm
    valid_node_norm = 1./((valid_graph.number_of_nodes())**0.5)
    valid_edge_norm = 1./((valid_graph.number_of_edges())**0.5)

    if use_cuda:
        valid_node_feat = valid_node_feat.cuda()
        valid_edge_feat = valid_edge_feat.cuda()
        valid_data = valid_data.cuda()
        valid_labels = valid_labels.cuda()

    model.eval()
    print("start eval")
    with torch.no_grad():
        pred = model(valid_graph, valid_node_feat, valid_edge_feat,
                        valid_node_norm,valid_edge_norm, valid_data)
        loss = model.get_loss(pred, valid_labels)
        print("Epoch {:04d} | Loss {:.4f} |".format(epoch, loss.item()))
