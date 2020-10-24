# import torch
# import numpy as np
# import torch.nn as nn
# from model.GATED_MLP import GatedGCN_MLP
#
#
# def get_mrr(pred, labels, hits = []):
#     soft = nn.Softmax(dim=1)
#     scores = soft(pred)
#     _, indices = torch.sort(scores, descending=True)
#     rank = torch.where((indices == labels))[1] + 1
#
#     #MRR
#     mrr = torch.mean(1.0/rank.float())
#     print('MRR: {}'.format(mrr))
#
#     #Hits
#     for hit in hits:
#         avg_count = torch.mean((rank <= hit).float())
#         print("Hits @ {}: {:.6f}".format(hit,avg_count.item()))
#
#
# def load_model(checkpoint_path):
#     net = GatedGCN_MLP(*args)
#     checkpoint = torch.load(checkpoint_path)
#     net.load_state_dict(checkpoint)
#     net.eval()
#
#     return net
#
# def inference(path):
#     model = load_net(path)
#
