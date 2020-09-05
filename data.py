import os
import numpy as np
import torch
from utilities import io

class LinkDataset(object):
    def __init__(self, name):
        self.name = name
        self.data_dir = os.path.join('data', self.name)
        self.relation_path = os.path.join(self.data_dir, 'relations.dict')
        self.entity_path = os.path.join(self.data_dir, 'entities.dict')


    def load_data(self):
        train_path = os.path.join(self.data_dir, 'train.txt')
        valid_path = os.path.join(self.data_dir, 'valid.txt')
        test_path = os.path.join(self.data_dir, 'test.txt')
        # Map entity and relation to integers
        entity_dict = io.read_dict(self.entity_path)
        relation_dict = io.read_dict(self.relation_path)
        # Return triplets as array
        self.train = np.asarray(io.read_trip_lst(train_path, entity_dict, relation_dict))
        self.valid = np.asarray(io.read_trip_lst(valid_path, entity_dict, relation_dict))
        self.test = np.asarray(io.read_trip_lst(test_path, entity_dict, relation_dict))
        #Return number of nodes and relation
        self.num_nodes = len(entity_dict)
        self.num_rels = len(relation_dict)
        print("# entities: {}".format(self.num_nodes))
        print("# relations: {}".format(self.num_rels))


if __name__ == '__main__':
    data = LinkDataset('FB15k-237')
    data.load_data()
    print(len(data.train))
    print(len(data.valid))
    print(len(data.test))
