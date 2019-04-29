import random
import os
import numpy as np
import socket
import torch
from scipy import misc
import pickle

class KTH(object):

    def __init__(self, train, data_root, seq_len = 20, image_size=64, data_type='drnet', pose=False):
        self.data_root = os.path.join(data_root, "KTH", "processed")
        self.seq_len = seq_len
        self.data_type = data_type
        self.image_size = image_size
        self.classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

        self.dirs = os.listdir(self.data_root)
        if train:
            self.train = True
            data_type = 'train'
            self.persons = list(range(1, 21))
        else:
            self.train = False
            self.persons = list(range(21, 26))
            data_type = 'test'

        self.data= {}
        self.pose = pose
        if pose:
            meta_file = 'pose'
            self.kp_mask = np.arange(51) % 3 != 2
            print('using external pose')
        else:
            meta_file = 'meta'
        for c in self.classes:
            filename = os.path.join(self.data_root, c, f"{data_type}_{meta_file}{image_size}x{image_size}.pkl")
            with open(filename, 'rb') as fi:
                self.data[c] = pickle.load(fi)


        self.seed_set = False

    def get_sequence(self):
        t = self.seq_len
        while True: # skip seqeunces that are too short
            c_idx = np.random.randint(len(self.classes))
            c = self.classes[c_idx]
            vid_idx = np.random.randint(len(self.data[c]))
            vid = self.data[c][vid_idx]
            seq_idx = np.random.randint(len(vid['files']))
            if len(vid['files'][seq_idx]) - t >= 0:
                break
        dname = '%s/%s/%s' % (self.data_root, c, vid['vid'])
        st = random.randint(0, len(vid['files'][seq_idx])-t)


        seq = []
        pose = []
        for i in range(st, st+t):
            fname = '%s/%s' % (dname, vid['files'][seq_idx][i])
            im = misc.imread(fname)/255.
            seq.append(im)
            if self.pose:
                pose.append(self.get_pose_code(vid['poses'][seq_idx][i]))
        if self.pose:
            return np.array(seq), np.stack(pose)
        else:
            return np.array(seq)
        return np.array(seq)

    # to speed up training of drnet, don't get a whole sequence when we only need 4 frames
    # x_c1, x_c2, x_p1, x_p2
    def get_drnet_data(self):
        c_idx = np.random.randint(len(self.classes))
        c = self.classes[c_idx]
        vid_idx = np.random.randint(len(self.data[c]))
        vid = self.data[c][vid_idx]
        seq_idx = np.random.randint(len(vid['files']))
        dname = '%s/%s/%s' % (self.data_root, c, vid['vid'])
        seq_len = len(vid['files'][seq_idx])

        seq = []
        pose = []
        for i in range(4):
            t = np.random.randint(seq_len)
            fname = '%s/%s' % (dname, vid['files'][seq_idx][t])
            im = misc.imread(fname)/255.
            seq.append(im)
            if self.pose:
                pose.append(self.get_pose_code(vid['poses'][seq_idx][t]))
        if self.pose:
            return np.array(seq), np.stack(pose)
        else:
            return np.array(seq)

    def get_pose_code(self, pose_dict):
        if pose_dict is None:
            kp = np.ones(35)
        else:
            kp = np.array(pose_dict['keypoints'])
            kp = (kp[self.kp_mask] / (self.image_size / 2)) - 1
            kp = np.concatenate([[0,], kp])
        return kp

    @staticmethod
    def totensor(tensor):
        if isinstance(tensor, tuple):
            return tuple([torch.from_numpy(t) for t in tensor])
        else:
            return torch.from_numpy(t)

    def __getitem__(self, index):
        if not self.seed_set:
            self.seed_set = True
            random.seed(index)
            np.random.seed(index)
            #torch.manual_seed(index)
        if not self.train or self.data_type == 'sequence':
            return self.totensor(self.get_sequence())
        elif self.data_type == 'drnet':
            return self.totensor(self.get_drnet_data())
        else:
            raise ValueError('Unknown data type: %d. Valid type: drnet | sequence.' % self.data_type)

    def __len__(self):
        return len(self.dirs)*36*5 # arbitrary

