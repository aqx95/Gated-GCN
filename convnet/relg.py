import torch
import torch.nn as nn
import torch.nn.functional as F

class RELGLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, num_bases, dropout, graph_norm, batch_norm, residual=False):
        super().__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.dropout = dropout
        self.graph_norm  = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual

        if in_dim != out_dim:
            self.residual = False;

        self.A = nn.Linear(in_dim, out_dim, bias=True)
        self.B = nn.Linear(in_dim, out_dim, bias=True)
        self.C = nn.Linear(in_dim, out_dim, bias=True)
        self.D = nn.Linear(in_dim, out_dim, bias=True)
        self.E = nn.Linear(in_dim, out_dim, bias=True)


        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.input_dim, self.output_dim))
        if self.num_bases < self.num_rels:
            # linear combination coefficients
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))

        #Batch norm
        self.bn_node_h = nn.BatchNorm1d(out_dim)
        self.bn_node_e = nn.BatchNorm1d(out_dim)



    def message_func(self,edges):
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.input_dim * self.output_dim)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.input_dim, self.output_dim)
        else:
            weight = self.weight

        # calculate msg @ W_r before put msg into edge
        # if src is torch.int64 we expect it is an index select
        etypes = torch.unique(edges.data['type'])
        msg = torch.empty((edges.src['h'].shape[0], self.output_dim),
                       device=edges.src['h'].device)
        for etype in etypes:
            loc = edges.data['type'] == etype
            w = weight[etype]
            src = edges.src['h'][loc]
            sub_msg = torch.matmul(src, w)
            msg[loc] = sub_msg

        Bh_j = edges.src['Bh']
        e_ij = edges.data['Ce'] + msg + edges.dst['Eh']
        edges.data['e'] = e_ij

        return {'Bh_j': Bh_j, 'e_ij': e_ij}

    def reduce_func(self, nodes):
        Ah_i = nodes.data['Ah']
        Bh_j = nodes.mailbox['Bh_j']
        e = nodes.mailbox['e_ij']
        sigma_ij = torch.sigmoid(e)
        h = Ah_i + torch.sum(sigma_ij*Bh_j, dim=1)/ (torch.sum(sigma_ij,dim=1) + 1e-6)
        return {'h': h}

    def forward(self, g, h, e, etype):
        h_in = h
        e_in = e
        g.edata['type'] = etype

        g.ndata['h'] = h
        g.ndata['Ah'] = self.A(h)
        g.ndata['Bh'] = self.B(h)
        #g.ndata['Dh'] = self.D(h)
        g.ndata['Eh'] = self.E(h)
        g.edata['Ce'] = self.C(e)
        g.edata['e'] = e

        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h'] #result of convolution
        e = g.edata['e'] #result of convolution

        norm_n = 1./((g.number_of_nodes())**0.5)
        norm_e = 1./((g.number_of_edges())**0.5)

        if self.graph_norm:
            h = h* norm_n
            e = e* norm_e

        if self.batch_norm:
            h = self.bn_node_h(h)
            e = self.bn_node_e(e)

        h = F.relu(h)
        e = F.relu(e)

        if self.residual:
            h = h_in + h
            e = e_in + e


        h = F.dropout(h, self.dropout, training = self.training)
        e = F.dropout(e, self.dropout, training = self.training)

        return h, e
