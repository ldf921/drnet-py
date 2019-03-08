import random
import os
import numpy as np
import socket
import torch
from scipy import misc
# from torch.utils.serialization import load_lua
import pickle
import re

class UCF:
    def __init__(self, train, data_root, seq_len = 20, image_size=64, split=1, data_type='drnet'):
        self.data_root = '%s/UCF101/processed/' % data_root
        self.seq_len = seq_len
        self.data_type = data_type
        self.image_size = image_size

        self.dirs = os.listdir(self.data_root)
        if train:
            self.train = True
            data_type = 'train'
        else:
            self.train = False
            data_type = 'test'

        self.max_dist = 80

        self.seed_set = False
        with open(os.path.join(data_root, 'UCF101', 'frames_%dx%d' % (image_size, image_size), '%s%d.pkl' % (data_type, split)), 'rb') as fi:
            self.data = pickle.load(fi)

    def get_sequence(self):
        t = self.seq_len
        while True: # skip seqeunces that are too short
            vid_idx = np.random.randint(len(self.data))
            vid = self.data[vid_idx]
            seq = vid['files']
            dname = '%s/%s' % (self.data_root, vid['vid'])
            seq_len = len(seq)
            if seq_len - t >= 0:
                break
        st = random.randint(0, seq_len -t)

        frames = []
        for i in range(st, st+t):
            fname = '%s/%s' % (dname, seq[i])
            im = misc.imread(fname)/255.
            frames.append(im)
        return np.array(frames)

    # to speed up training of drnet, don't get a whole sequence when we only need 4 frames
    # x_c1, x_c2, x_p1, x_p2
    def get_drnet_data(self):
        vid_idx = np.random.randint(len(self.data))
        vid = self.data[vid_idx]
        seq = vid['files']
        dname = '%s/%s' % (self.data_root, vid['vid'])
        if len(seq) >= self.max_dist:
            st = random.randint(0, len(seq) - self.max_dist)
            seq = seq[st : st + self.max_dist]

        seq_len = len(seq)

        frames = []
        for i in range(4):
            t = np.random.randint(seq_len)
            fname = '%s/%s' % (dname, seq[t])
            im = misc.imread(fname)/255.
            frames.append(im)
        return np.array(frames)

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            #torch.manual_seed(index)
        if not self.train or self.data_type == 'sequence':
            return torch.from_numpy(self.get_sequence())
        elif self.data_type == 'drnet':
            return torch.from_numpy(self.get_drnet_data())
        else:
            raise ValueError('Unknown data type: %d. Valid type: drnet | sequence.' % self.data_type)

    def __len__(self):
        return len(self.dirs)*36*5 # arbitrary

