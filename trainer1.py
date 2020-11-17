import os
import numpy as np
import torch.nn as nn
import torch
from glob import glob
import dgl
from tqdm import tqdm

class Fitter:
    def __init__(self, net, config, device):
        self.model = net
        self.config = config
        self.device = device

        self.base_dir = self.config['train']['folder']
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

        self.best_loss = 10**5
        self.hist_loss = []
        self.val_loss = []
        self.epoch = 0
        self.log_path = f'{self.base_dir}/log.txt'

        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['optimizer']['lr'],
                                          weight_decay=config['optimizer']['regularization'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.1)

        self.log('Begin training with {}'.format(self.device))
        self.model = self.model.to(self.device)


    def fit(self, graph_data, test_graph, valid_data, valid_labels):
        self.hist_loss = []
        for i in tqdm(range(self.config['train']['n_epochs'])):
            train_loss = self.train_epoch(graph_data)
            self.hist_loss.append(train_loss)
            self.log(f'[TRAINING] Epoch {self.epoch}    Loss: {train_loss}')

            # valid_loss = self.validate_epoch(test_graph, valid_data, valid_labels)
            # self.val_loss.append(valid_loss)
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
            self.scheduler.step()
            self.epoch += 1

        return self.hist_loss

    def train_epoch(self, graph_data):
        self.model = self.model.to(self.device)
        self.model.train()
        g, node_id, edge_type, data, labels = graph_data.prep_train_graph(self.config['graph_obj']['batch_size'],
                                              self.config['graph_obj']['split_size'],
                                              self.config['graph_obj']['neg_sampling'],
                                              self.config['graph_obj']['edge_sampler'])

        node_id, edge_type = node_id.to(self.device), edge_type.to(self.device)
        g, data, labels = g.to(self.device), data.to(self.device), labels.to(self.device)
        embeddings = self.model(g, node_id, edge_type)
        train_loss = self.model.get_loss(embeddings, data, labels)

        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['optimizer']['grad_norm'])
        self.optimizer.step()
        self.optimizer.zero_grad()

        del g, embeddings, data, labels

        return train_loss.detach().item()


    def validate_epoch(self, test_graph, valid_data, valid_labels):
        self.model = self.model.to('cpu')
        self.model.eval()

        with torch.no_grad():
            valid_embed = self.model(test_graph, valid_data)
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
