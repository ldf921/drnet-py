from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from shutil import copyfile
from tqdm import tqdm
from typing import Tuple

import valid
from utils import utils
from utils.metrics import Summary
from models.gan import DrGan
from models.cgan import CGan, CGanTriplet


def random_data(n=4, b=1):
    x = [torch.randn(b, 3, 128, 128) for _ in range(n)]
    p = [torch.rand(b, 35) * 2 - 1 for _ in range(n)]
    return x, p

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=50, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--data_root', default='', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600 * 100, help='number of samples per epoch')
parser.add_argument('--content_dim', type=int, default=128, help='size of the content vector')
parser.add_argument('--pose_dim', type=int, default=10, help='size of the pose vector')
parser.add_argument('--image_width', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='kth', help='dataset to train with')
parser.add_argument('--max_step', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--sd_weight', type=float, default=0.0001, help='weight on adversarial loss')
parser.add_argument('--sd_nf', type=int, default=100, help='number of layers')
parser.add_argument('--content_model', default='dcgan_unet', help='model type (dcgan | dcgan_unet | vgg_unet)')
parser.add_argument('--pose_model', default='dcgan', help='model type (dcgan | unet | resnet)')
parser.add_argument('--data_threads', type=int, default=5, help='number of parallel data loading threads')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('--data_type', default='drnet', help='speed up data loading for drnet training')
parser.add_argument('--pose', action='store_true', help='use the extracted pose code')
parser.add_argument('--test', action='store_true', help='test the saved checkpoints')
parser.add_argument('--saveimg', action='store_true', help='store_images')
parser.add_argument('--saveidx', default=None, type=str)
parser.add_argument('--checkpoint', default=None, type=str, help='the file name of checkpoint (model.pth)')
parser.add_argument('--swap_loss', default=None, type=str)

opt = parser.parse_args()
model = CGanTriplet(opt)
model.build_optimizer()
model.train(random_data(b=4))
