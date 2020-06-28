"""
Dataloader for the MetaChrom model
"""
__author__ = "Ben Lai"
__copyright__ = "Copyright 2020, TTIC"
__license__ = "GNU GPLv3"


import numpy as np
import torch
from torch.utils import data as D

def seq2onehot(seq):
    window_size = 1000
    matrix = np.zeros(shape = (window_size, 4), dtype = np.uint8)
    for i, nt in enumerate(seq):
        if nt == "A":
            matrix[i][0] = 1
        elif nt == "G":
            matrix[i][1] = 1
        elif nt == "C":
            matrix[i][2] = 1
        elif nt == "T":
            matrix[i][3] = 1
        else:
            continue
    return matrix

class seq_data(D.Dataset):
    def __init__(self, seq_path, training_mode = False ,label_path = None):
        self.seq_path = seq_path
        self.seq_list = []
        self.training_mode = training_mode
        if self.training_mode:
            self.label_dict = torch.load(label_path)

        for line in open(self.seq_path, 'r'):
            line = line.split('\t')
            self.seq_list.append((line[0], line[1]))
        self.len = len(self.seq_list)

    def __getitem__(self, index):
        sample = self.seq_list[index]
        seq = sample[1]
        id = sample[0]
        oh_seq = seq2onehot(seq)
        oh_seq = torch.from_numpy(oh_seq.T).float()
        if self.training_mode:
            label = self.label_dict[id]
            label = torch.from_numpy(label).float()
            return id, oh_seq, label
        else:
            return id, oh_seq

    def __len__(self):
        return(self.len)


class SNP_data(D.Dataset):
    def __init__(self, seq_path):
        seq_file = open(seq_path,'r')
        self.seq_list = []
        for line in seq_file:
            self.seq_list.append(line.strip().split('\t'))
        self.len = len(self.seq_list)

    def __getitem__(self,index):
        seqs = self.seq_list[index]
        ref_seq = seqs[1]
        alt_seq = seqs[2]
        id = seqs[0]
        ref_seq = torch.from_numpy(seq2onehot(ref_seq).T).float()
        alt_seq = torch.from_numpy(seq2onehot(alt_seq).T).float()
        return id, ref_seq, alt_seq

    def __len__(self):
        return self.len


if __name__ == '__main__':
    Dset = seq_data(seq_path='../../data/seq_files/test.seq', label_path='../../data/seq_files/labels.pt')
    for seq in Dset:
        print(seq[0].size())
        print(seq[1].size())
        break
    
    Dset = SNP_data(seq_path='../../data/SNP_test/out.seq')
    for seq in Dset:
        print(seq[0].size())
        print(seq[1].size())
        break