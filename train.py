import os
import pandas as pd
import random
from data import LinkDataset
import utilities.utils as utils
import numpy as np
import torch
from graphdata import DGLData
from model.RGCN import LinkPredict
from model.GATEDGCN import GatedGCN
from model.GATED_MLP import GatedGCN_MLP
import timezz
import utilities.evaluator as eval
import matplotlib.pyplot as plt

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)  # set PYTHONHASHSEED env var at fixed value
    random.seed(seed)  #set fixed value for python built-in pseudo-random generator
    np.random.seed(seed) # for numpy pseudo-random generator
    torch.manual_seed(seed) # pytorch (both CPU and CUDA)

set_seed(2020)

class args:
    dropout = 0.5
    n_hidden = 500
    gpu = 0
    lr = 0.01
    n_bases = 100
    n_layers = 2
    n_epochs = 2
    dataset='FB15k-237'
    eval_batch_size = 500
    eval_protocol = 'filtered'
    regularization = 0.01
    grad_norm = 1.0
    graph_batch_size = 30000
    graph_split_size = 0.5
    negative_sample = 5
    eval_every= 30
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
test_graph, test_rel, test_norm = test_dgl.prepare_test()
test_deg = test_graph.in_degrees(
            range(test_graph.number_of_nodes())).float().view(-1,1)
test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
test_rel = torch.from_numpy(test_rel)
test_norm = utils.node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))



train_dgl = DGLData(train_data, num_nodes, num_rels)


model1 =  LinkPredict(num_nodes,
                    args.n_hidden,
                    num_rels,
                    num_bases=args.n_bases,
                    num_hidden_layers=args.n_layers,
                    dropout=args.dropout,
                    use_cuda=use_cuda,
                    reg_param=args.regularization)

model2 = GatedGCN(num_nodes,
                in_dim_edge=num_rels,
                hid_dim=args.n_hidden,
                out_dim=args.n_hidden,
                n_hidden_layers=args.n_layers,
                dropout=args.dropout,
                graph_norm=True,
                batch_norm=True,
                residual=True)


model3 = GatedGCN_MLP(num_nodes,
                in_dim_edge=num_rels,
                hid_dim=args.n_hidden,
                out_dim=args.n_hidden,
                n_hidden_layers=args.n_layers,
                dropout=args.dropout,
                graph_norm=True,
                batch_norm=True,
                residual=True)

mod_lst = {'model1':model1, 'model2':model2, 'model3':model3}
loss_df = [[int(i+1) for i in range(args.n_epochs)]]
for name, models in mod_lst.items():
    model = models
    training_loss = []
    if use_cuda:
        model.cuda()

    # optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.1,
                                                           patience=10, min_lr=1e-5)

    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    for epoch in range(args.n_epochs):
        model.train()
        epoch += 1

        g, node_id, edge_type, node_norm, data, labels =train_dgl.prepare_train(30000,0.5,10)

        node_id = torch.from_numpy(node_id)
        edge_type = torch.from_numpy(edge_type)
        edge_norms = utils.node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
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
        if name == 'model1':
            embed, pred = model(g, node_id.view(-1, 1).long(), edge_type, edge_norms, data)
        else:
            embed, pred = model(g, node_feat, edge_feat, node_norm, edge_norm, data)
        loss = model.get_loss(embed, pred.squeeze(), labels)
        training_loss.append(loss.detach().item())
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

    loss_df.append(training_loss)

loss_dfs = pd.DataFrame.from_records(loss_df).transpose()
loss_dfs.columns = ['Epochs','RGCN','GATED', 'GATEDMLP']
loss_dfs.Epochs = loss_dfs.Epochs.astype('int32')
print(loss_df)
plt.plot( 'Epochs', 'RGCN', color='red', data=loss_dfs, label='RGCN')
plt.plot( 'Epochs', 'GATED', color='blue', data=loss_dfs, label='GATED')
plt.plot( 'Epochs', 'GATEDMLP', color='green', data=loss_dfs, label='GATEDMLP')
plt.title('Training Loss against epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig('training_loss.png')
    #del g, node_feat, edge_feat, edge_norm, embed

    # # validation
    # if epoch % args.eval_every == 0:
    #     #Set node and edge features
    #     test_node_feat = np.zeros((test_graph.number_of_nodes(), num_nodes))
    #     test_node_feat[np.arange(test_graph.number_of_nodes()), test_node_id] = 1.0
    #     test_node_feat = torch.FloatTensor(test_node_feat)
    #
    #     test_edge_feat = np.zeros((test_graph.number_of_edges(), num_rels))
    #     test_edge_feat[np.arange(test_graph.number_of_edges()), test_rel] = 1.0
    #     test_edge_feat = torch.FloatTensor(test_edge_feat)
    #
    #     #norm
    #     node_norm = 1./((test_graph.number_of_nodes())**0.5)
    #     edge_norm = 1./((test_graph.number_of_edges())**0.5)
    #
    #     model.eval()
    #     print("start eval")
    #     with torch.no_grad():
    #         embed,e = model(test_graph, test_node_feat, test_edge_feat,1,1)
    #         mrr = eval.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
    #                              valid_data, test_data, hits=[1, 3, 10], eval_bz=args.eval_batch_size,
    #                              eval_p=args.eval_protocol)
