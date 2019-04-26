import torch
import socket
import numpy as np
import scipy.misc
import functools
from torch.autograd import Variable
from torch.utils.data import DataLoader

from data.moving_mnist import MovingMNIST
from data.kth import KTH
from data import suncg

class NormalizedDataLoader(DataLoader):
    def __init__(self, dataloader, opt):
        self.dataloader = dataloader
        self.opt = opt

    def __iter__(self):
        for data in self.dataloader:
            data = normalize_data(self.opt, torch.cuda.FloatTensor, data)
            yield data

    def __len__(self):
        return len(self.dataloader)


def load_data(opt):
    if opt.dataset == 'moving_mnist':
        train_data = MovingMNIST(
                train=True,
                data_root=opt.data_root,
                seq_len=opt.max_step,
                image_size=opt.image_width,
                num_digits=2)
        test_data = MovingMNIST(
                train=False,
                data_root=opt.data_root,
                seq_len=opt.max_step,
                image_size=opt.image_width,
                num_digits=2)
    elif opt.dataset == 'suncg':
        train_data = suncg.SUNCG(
                train=True,
                data_root=opt.data_root,
                seq_len=opt.max_step,
                image_size=opt.image_width)
        test_data = suncg.SUNCG(
                train=False,
                data_root=opt.data_root,
                seq_len=opt.max_step,
                image_size=opt.image_width)
    elif opt.dataset == 'kth':
        train_data = KTH(
                train=True,
                pose=opt.pose,
                data_root=opt.data_root,
                seq_len=opt.max_step,
                image_size=opt.image_width,
                data_type=opt.data_type)
        test_data = KTH(
                train=False,
                pose=opt.pose,
                data_root=opt.data_root,
                seq_len=opt.max_step,
                image_size=opt.image_width,
                data_type=opt.data_type)
    return train_data, test_data


def get_dataloader(opt):
    train_data, test_data = load_data(opt)
    train_loader = DataLoader(train_data,
                              num_workers=opt.data_threads,
                              batch_size=opt.batch_size,
                              shuffle=True,
                              drop_last=True,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             num_workers=opt.data_threads,
                             batch_size=opt.batch_size,
                             shuffle=True,
                             drop_last=True,
                             pin_memory=True)
    return train_loader, test_loader


def get_normalized_dataloader(opt):
    train_loader, test_loader = get_dataloader(opt)
    train_loader = NormalizedDataLoader(train_loader, opt)
    test_loader = NormalizedDataLoader(test_loader, opt)
    return train_loader, test_loader


def sequence_input(seq, dtype):
    return [Variable(x.type(dtype)) for x in seq]


def normalize_data(opt, dtype, data):
    if isinstance(data, list):
        sequence, pose = data
        pose.transpose_(0, 1)
    else:
        sequence = data
    if opt.dataset == 'moving_mnist':
        sequence.transpose_(0, 1)
        if opt.channels > 1:
            sequence.transpose_(3, 4).transpose_(2, 3)
        else:
            sequence.unsqueeze_(2)
    elif opt.dataset == 'suncg' or opt.dataset == 'suncg_dual' or opt.dataset == 'kth':
        sequence.transpose_(0, 1)
        sequence.transpose_(3, 4).transpose_(2, 3)
    else:
        sequence.transpose_(0, 1)


    if isinstance(data, list):
        return sequence_input(sequence, dtype), sequence_input(pose, dtype)
    else:
        return sequence_input(sequence, dtype)


def is_sequence(arg):
    return (not hasattr(arg, "strip") and
            not type(arg) is np.ndarray and
            not hasattr(arg, "dot") and
            (hasattr(arg, "__getitem__") or
            hasattr(arg, "__iter__")))


def image_tensor(inputs, padding=1):
    # assert is_sequence(inputs)
    assert len(inputs) > 0
    # print(inputs)

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [image_tensor(x) for x in inputs]
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim * len(images) + padding * (len(images)-1),
                            y_dim)
        for i, image in enumerate(images):
            result[:, i * x_dim + i * padding :
                   (i+1) * x_dim + i * padding, :].copy_(image)

        return result

    # if this is just a list, make a stacked image
    else:
        images = [x.data if isinstance(x, torch.autograd.Variable) else x
                  for x in inputs]
        # print(images)
        if images[0].dim() == 3:
            c_dim = images[0].size(0)
            x_dim = images[0].size(1)
            y_dim = images[0].size(2)
        else:
            c_dim = 1
            x_dim = images[0].size(0)
            y_dim = images[0].size(1)

        result = torch.ones(c_dim,
                            x_dim,
                            y_dim * len(images) + padding * (len(images)-1))
        for i, image in enumerate(images):
            result[:, :, i * y_dim + i * padding :
                   (i+1) * y_dim + i * padding].copy_(image)
        return result


def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    # pdb.set_trace()
    return scipy.misc.toimage(tensor.numpy(),
                              high=255*tensor.max().item(),
                              channel_axis=0)


def save_image(filename, tensor):
    img = make_image(tensor)
    img.save(filename)


def save_tensors_image(filename, inputs, padding=1):
    images = image_tensor(inputs, padding)
    return save_image(filename, images)


def prod(l):
    return functools.reduce(lambda x, y: x * y, l)


def batch_flatten(x):
    return x.resize(x.size(0), prod(x.size()[1:]))


def clear_progressbar():
    # moves up 3 lines
    print("\033[2A")
    # deletes the whole line, regardless of character position
    print("\033[2K")
    # moves up two lines again
    print("\033[2A")


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

