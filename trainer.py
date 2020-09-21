import os
import numpy as np
import torch.nn as nn
import torch
from glob import glob

class Fitter:
    def __init__(self, net, config, device):
        self.model = net
        self.config = config
        self.device = device

        self.base_dir = self.config['train']['folder']
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.best_loss = 10**5
        self.epoch = 0
        self.log_path = f'{self.base_dir}/log.txt'

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['optimizer']['lr'],
                                          weight_decay=config['optimizer']['regularization'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=0.1,patience=3)

        self.log('Begin training with {}'.format(self.device))


    def fit(self, graph_data, test_graph, valid_data, test_labels, valid_labels):
        for i in range(self.config['train']['n_epochs']):
            train_loss = self.train_epoch(graph_data)
            self.log(f'[TRAINING] Epoch {self.epoch}    Loss: {train_loss}')

            # valid_loss = self.validate_epoch(test_graph, valid_data, valid_labels)
            # self.log(f'[VALIDATION] Epoch {self.epoch}    Loss: {valid_loss}')
            #
            # if self.best_loss > valid_loss:
            #     self.best_loss = valid_loss
            #     self.model.eval()
            #     self.save(f'{self.base_dir}/best-checkpoint-epoch{str(self.epoch).zfill(2)}.bin')
            #     for path in sorted(glob(f'{self.base_dir}/best-checkpoint-epoch*.bin'))[:-3]:
            #         os.remove(path)
            #
            # self.scheduler.step(metrics=valid_loss)
            self.epoch += 1


    def train_epoch(self, graph_data):
        self.model.train()
        g, data = graph_data.prep_train_graph(self.config['graph_obj']['batch_size'],
                                              self.config['graph_obj']['split_size'],
                                              self.config['graph_obj']['neg_sampling'],
                                              self.config['graph_obj']['edge_sampler'])

        labels = torch.LongTensor(data[:,1])
        g, labels = g.to(self.device), labels.to(self.device)
        #norm
        node_norm = 1./((g.number_of_nodes())**0.5)
        edge_norm = 1./((g.number_of_edges())**0.5)

        train_pred = self.model(g, node_norm, edge_norm, data)
        train_loss = self.model.get_loss(train_pred, labels)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['optimizer']['grad_norm'])
        self.optimizer.step()
        self.optimizer.zero_grad()

        del g, data, labels

        return train_loss.detach().item()


    def validate_epoch(self, test_graph, valid_data, valid_labels):
        self.model.eval()

        #norm
        valid_node_norm = 1./((test_graph.number_of_nodes())**0.5)
        valid_edge_norm = 1./((test_graph.number_of_edges())**0.5)

        valid_data, valid_labels = valid_data.to(self.device), valid_labels.to(self.device)

        with torch.no_grad():
            valid_pred = self.model(test_graph, valid_node_norm, valid_edge_norm, valid_data)
            valid_loss = self.model.get_loss(valid_pred, valid_labels)

        return valid_loss.detach().item()


    def save(self, path):
        self.model.eval()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_summary_loss': self.best_loss,
            'epoch': self.epoch}, path)


    def log(self, message):
        if self.config['train']['verbose']:
            print(message)
        with open(self.log_path, 'a+') as logger:
            logger.write(f'{message}\n')
