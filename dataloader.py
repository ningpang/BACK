import torch
import torch.utils.data as data
import os
import numpy as np
import random
import json

def get_word_vec(args, base_dir):
    word_path = base_dir + '/' + 'glove.6B.50d.json'
    if not os.path.exists(word_path):
        print("[Error] Data file does not exists !")
        assert 0
    word_vec = json.load(open(word_path, 'r'))
    word_total = len(word_vec)
    word_dim = len(word_vec[0]['vec'])
    word_vec_mat = np.zeros((word_total, word_dim), dtype=np.float32)
    word2id = {}
    for cur_id, word in enumerate(word_vec):
        w = word['word'].lower()
        word2id[w] = cur_id
        word_vec_mat[cur_id, :] = word['vec']
        word_vec_mat[cur_id] =word_vec_mat[cur_id] / np.sqrt((np.sum(word_vec_mat[cur_id] ** 2)))

    UNK = word_total
    PAD = word_total + 1
    word2id['UNK'] = UNK
    word2id['PAD'] = PAD
    return word2id, word_vec_mat



class FTCDataset(data.Dataset):
    def __init__(self, args, base_dir, state, word2id):
        self.args = args
        self.data_path = base_dir+'/'+args.dataset+'/'+state+'.json'
        self.word_path = base_dir+'/'+'glove.6B.50d.json'
        if not os.path.exists(self.data_path):
            print("[Error] Data file does not exists !")
            assert 0
        self.data = json.load(open(self.data_path, 'r'))
        self.word2id = word2id

        self.classes = list(self.data.keys())
        if state == 'train':
            self.N = args.N_for_train
        else:
            self.N = args.N # classes
        self.K = args.K # support ins
        self.Q = args.Q # query


    def __getraw__(self, item):
        word, mask = self.tokenize(item['tokens'])
        return word, mask

    def __additem__(self, d, word, mask):
        d['word'].append(word)
        d['mask'].append(mask)

    def __getitem__(self, item):
        target_classes = random.sample(self.classes, self.N)
        support_set = {'word':[], 'mask':[]}
        query_set = {'word':[], 'mask':[]}
        query_label = []
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(list(range(len(self.data[class_name]))),
                                       self.K+self.Q, False)
            count = 0
            for j in indices:
                word, mask = self.__getraw__(self.data[class_name][j])
                word = torch.tensor(word).long()
                mask = torch.tensor(mask).long()
                if count < self.K:
                    self.__additem__(support_set, word, mask)
                else:
                    self.__additem__(query_set, word, mask)
                count += 1
            query_label += [i]*self.Q
        return support_set, query_set, query_label

    def tokenize(self, tokens):
        idx_tokens = []
        mask = []
        max_length = self.args.max_length
        for tk in tokens:
            tk = tk.lower()
            if tk in self.word2id:
                idx_tokens.append(self.word2id[tk])
            else:
                idx_tokens.append(self.word2id['UNK'])
        mask += len(idx_tokens)*[1]
        if len(idx_tokens)<max_length:
            mask += (max_length - len(idx_tokens)) * [0]
            idx_tokens += (max_length-len(idx_tokens))*[self.word2id['PAD']]

        else:
            idx_tokens = idx_tokens[:max_length]
            mask = mask[:max_length]

        assert len(idx_tokens) == len(mask) == max_length

        return idx_tokens, mask

    def __len__(self):
        return 10000000


def collate_fn(data):
    batch_support = {'word':[], 'mask':[]}
    batch_query = {'word': [], 'mask': []}
    batch_label = []
    support_sets, query_sets, query_labels = zip(*data)
    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k] += support_sets[i][k]
        for k in query_sets[i]:
            batch_query[k] += query_sets[i][k]
        batch_label += query_labels[i]
    for k in batch_support:

        batch_support[k] = torch.stack(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.stack(batch_query[k], 0)
    batch_label = torch.tensor(batch_label)

    return batch_support, batch_query, batch_label

def get_loader(args, base_dir, state, word2id, num_workers=1, collate_fn=collate_fn):
    dataset = FTCDataset(args, base_dir, state, word2id)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  pin_memory=False,
                                  num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)

