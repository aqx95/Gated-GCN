import numpy as np
import dgl
import torch

# Retreive mapping for entities/relationd dict
def read_dict(filename):
    d = {}
    with open(filename,'r') as file:
        for line in file:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

# Map entities/relation to integers
def read_trip(filename):
    with open(filename, 'r') as file:
        for line in file:
            processed_line = line.strip().split('\t')
            yield processed_line

# Generate triplets
def read_trip_lst(filename, entity_dict, relation_dict):
    lst = []
    for triplet in read_trip(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        lst.append([s, r, o])
    return lst
