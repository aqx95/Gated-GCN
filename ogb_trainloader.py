# from torch.utils.data import Dataset
#
# class Trainset(Dataset):
#     def __init__(self, triples, nentity, nrelation, neg_sample_size, mode):
#         self.triples = triples
#         self.nentity = entity
#         self.nrelation = nrelation
#         self.neg_sample_size = neg_sample_size
#         self.mode = mode
#
#     def __len__(self):
#         return len(triples['head'])
#
#     def __getitem__(self, idx):
#         head, relation, tail = self.triples['head'], self.triples['relation'], self.triples['tail']
#         pos_sample = torch.LongTensor([head, relation, tail])
#         neg_sample = torch.randint(0, self.nentity, (self.neg_sample_size,))
#
#         return pos_sample, neg_sample, self.mode
#
#     @staticmethod
#     def collate_fn(data):
#         positive_sample = torch.stack([_[0] for _ in data], dim=0)
#         negative_sample = torch.stack([_[1] for _ in data], dim=0)
#         mode = data[0][2]
#         return positive_sample, negative_sample, mode
#
#
# class TestDataset(Dataset):
#     def __init__(self, triples, args, mode, random_sampling):
#         self.triples = triples
#         self.nentity = args.nentity
#         self.nrelation = args.nrelation
#         self.mode = mode
#         self.random_sampling = random_sampling
#         if random_sampling:
#             self.neg_size = args.neg_size_eval_train
#
#     def __len__(self):
#         return len(self.triples['head'])
#
#     def __getitem__(self, idx):
#         head, relation, tail = self.triples['head'][idx], self.triples['relation'][idx], self.triples['tail'][idx]
#         positive_sample = torch.LongTensor((head, relation, tail))
#
#         if self.mode == 'head-batch':
#             if not self.random_sampling:
#                 negative_sample = torch.cat([torch.LongTensor([head]), torch.from_numpy(self.triples['head_neg'][idx])])
#             else:
#                 negative_sample = torch.cat([torch.LongTensor([head]), torch.randint(0, self.nentity, size=(self.neg_size,))])
#         elif self.mode == 'tail-batch':
#             if not self.random_sampling:
#                 negative_sample = torch.cat([torch.LongTensor([tail]), torch.from_numpy(self.triples['tail_neg'][idx])])
#             else:
#                 negative_sample = torch.cat([torch.LongTensor([tail]), torch.randint(0, self.nentity, size=(self.neg_size,))])
#
#         return positive_sample, negative_sample, self.mode
#
#     @staticmethod
#     def collate_fn(data):
#         positive_sample = torch.stack([_[0] for _ in data], dim=0)
#         negative_sample = torch.stack([_[1] for _ in data], dim=0)
#         mode = data[0][2]
#
#         return positive_sample, negative_sample, mode
#
# class BidirectionalOneShotIterator(object):
#     def __init__(self, dataloader_head, dataloader_tail):
#         self.iterator_head = self.one_shot_iterator(dataloader_head)
#         self.iterator_tail = self.one_shot_iterator(dataloader_tail)
#         self.step = 0
#
#     def __next__(self):
#         self.step += 1
#         if self.step % 2 == 0:
#             data = next(self.iterator_head)
#         else:
#             data = next(self.iterator_tail)
#         return data
#
#     @staticmethod
#     def one_shot_iterator(dataloader):
#         '''
#         Transform a PyTorch Dataloader into python iterator
#         '''
#         while True:
#             for data in dataloader:
#                 yield data
