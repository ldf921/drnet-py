import torch
import itertools
from torch import nn as nn
from torch import optim
import numpy as np
import scipy.misc
import functools
from torch.autograd import Variable
from torch.utils.data import DataLoader
from typing import Tuple

from data.moving_mnist import MovingMNIST
from data.kth import KTH
from data import suncg


def get_optimizer(opt, model: nn.Module) -> optim.Optimizer:
    """
    :return: get a optimizer for each of the network with optimizer type, learning rate, and beta1 set as in `opt`
    """
    if opt.optimizer == 'adam':
        optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % opt.optimizer)

    return optimizer(model.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def get_initialized_network(opt) -> Tuple[nn.Module]:
    """
    :return: content and pose encoder, decoder and scene discriminator, with `utils.init_weights` applied
    """
    if opt.image_width == 64:
        import models.resnet_64 as resnet_models
        import models.dcgan_64 as dcgan_models
        import models.dcgan_unet_64 as dcgan_unet_models
        import models.vgg_unet_64 as vgg_unet_models
    elif opt.image_width == 128:
        import models.resnet_128 as resnet_models
        import models.dcgan_128 as dcgan_models
        import models.dcgan_unet_128 as dcgan_unet_models
        import models.vgg_unet_128 as vgg_unet_models
    import models.classifiers as classifiers

    # load models
    if opt.content_model == 'dcgan_unet':
        netEC = dcgan_unet_models.content_encoder(opt.content_dim, opt.channels)
        netD = dcgan_unet_models.decoder(opt.content_dim, opt.pose_dim, opt.channels)
    elif opt.content_model == 'vgg_unet':
        netEC = vgg_unet_models.content_encoder(opt.content_dim, opt.channels)
        netD = vgg_unet_models.decoder(opt.content_dim, opt.pose_dim, opt.channels)
    elif opt.content_model == 'dcgan':
        netEC = dcgan_models.content_encoder(opt.content_dim, opt.channels)
        netD = dcgan_models.decoder(opt.content_dim, opt.pose_dim, opt.channels)
    else:
        raise ValueError('Unknown content model: %s' % opt.content_model)

    if opt.pose_model == 'dcgan':
        netEP = dcgan_models.pose_encoder(opt.pose_dim, opt.channels, normalize=opt.normalize)
    elif opt.pose_model == 'resnet':
        netEP = resnet_models.pose_encoder(opt.pose_dim, opt.channels, normalize=opt.normalize)
    else:
        raise ValueError('Unknown pose model: %s' % opt.pose_model)
    netC = classifiers.scene_discriminator(opt.pose_dim, opt.sd_nf)

    netEC.apply(init_weights)
    netEP.apply(init_weights)
    netD.apply(init_weights)
    netC.apply(init_weights)

    return netEC, netEP, netD, netC


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
    """
    :return: raw data
    """
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
                epoch_samples=opt.epoch_size,
                pose=opt.pose,
                data_root=opt.data_root,
                seq_len=opt.max_step,
                image_size=opt.image_width,
                data_type=opt.data_type)
        test_data = KTH(
                train=False,
                epoch_samples=opt.epoch_size,
                pose=opt.pose,
                data_root=opt.data_root,
                seq_len=opt.max_step,
                image_size=opt.image_width,
                data_type=opt.data_type)
    return train_data, test_data


def get_dataloader(opt):
    """
    :return: construct data loader directly from raw data
    """
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
                             shuffle=False,
                             drop_last=True,
                             pin_memory=True)
    print(len(train_data), len(test_data), len(train_loader), len(test_loader))
    return train_loader, test_loader


def get_normalized_dataloader(opt):
    """
    :return: get normalized data from raw data by applying normalizing transformations
    """
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


def tensor_seq_to_tensor(inputs, padding=1) -> torch.tensor:
    assert len(inputs) > 0

    # if this is a list of lists, unpack them all and grid them up
    if is_sequence(inputs[0]) or (hasattr(inputs, "dim") and inputs.dim() > 4):
        images = [tensor_seq_to_tensor(x) for x in inputs]
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
            result[:, :, i * y_dim + i * padding:(i+1) * y_dim + i * padding].copy_(image)
        return result


def tensor_seq_to_img(inputs, padding=1):
    tensor = tensor_seq_to_tensor(inputs, padding)
    tensor = tensor.cpu().clamp(0, 1)
    if tensor.size(0) == 1:
        tensor = tensor.expand(3, tensor.size(1), tensor.size(2))
    img = scipy.misc.toimage(tensor.numpy(),
                             high=255*tensor.max().item(),
                             channel_axis=0)
    return img


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


def print_write_log(msg, f):
    print(msg)
    print(msg, file=f)
    f.flush()


# --------- plotting functions ------------------------------------
def plot_rec(pose, models, x, max_step):
    try:
        netEC, netEP, netD, _ = models
    except:
        netEC = models.netEC
        netD = models.netD
    if pose:
        x, p = x
        x_c = x[0]
        h_c = netEC(x_c)
        t = np.random.randint(1, max_step)
        x_p = x[t]
        h_p = p[t]
        h_p = h_p.unsqueeze(2).unsqueeze(3)
    else:
        x_c = x[0]
        x_p = x[np.random.randint(1, max_step)]

        h_c = netEC(x_c)
        h_p = netEP(x_p)

    rec = netD([h_c, h_p])
    x_c, x_p, rec = x_c.data, x_p.data, rec.data
    to_plot = []
    row_sz = 5
    nplot = 20
    for i in range(0, nplot-row_sz, row_sz):
        row = [[xc, xp, xr] for xc, xp, xr in zip(x_c[i:i+row_sz], x_p[i:i+row_sz], rec[i:i+row_sz])]
        to_plot.append(list(itertools.chain(*row)))

    img = tensor_seq_to_img(to_plot)
    return img


def plot_analogy(pose, models, x, channels, image_width, max_step):
    try:
        netEC, netEP, netD, _ = models
    except:
        netEC = models.netEC
        netD = models.netD
    if pose:
        x, p = x
    x_c = x[0]

    h_c = netEC(x_c)
    nrow = 10
    row_sz = max_step
    to_plot = []
    row = [xi[0].data for xi in x]  # first batch every frame
    zeros = torch.zeros(channels, image_width, image_width)
    to_plot.append([zeros] + row)
    for i in range(nrow):
        to_plot.append([x[0][i].data])  # first frame every batch

    for j in range(0, row_sz):
        if pose:
            h_p = p[j].unsqueeze(2).unsqueeze(3)
        else:
            h_p = netEP(x[j])
        for i in range(nrow):
            h_p[i] = h_p[0]
        rec = netD([h_c, Variable(h_p)])
        for i in range(nrow):
            to_plot[i+1].append(rec[i].data.clone())

    img = tensor_seq_to_img(to_plot)
    return img


def plot_reconstr(pose: bool, models: Tuple[torch.nn.Module],
                   original: torch.tensor, content: torch.tensor, num_frame=10, repeat_cont=True):
    """
    pose True:
        x, x_c: (T of [1, C, H, W], T of [1, 35]) or ([T, 1, C, H, W], [T, 1, 35])
    pose False:
        x, x_c: T of [1, C, H, W] or [T, 1, C, H, W]
    :param pose: whether we have pretrained pose code
    :param models: tuple of content encoder, pose encoder, decoder and scence discriminator
    :param original: original video
    :param content: content video
    :param num_frame: maximum frames (i.e. # of columns)
    :param repeat_cont: whether repeat the same content code
    :return: (T, 3, H, W)
    """
    netEC, netEP, netD, _ = models
    if pose:
        original, original_p = original
        content, _ = content
        original, original_p = original[:num_frame], original_p[:num_frame]
        content = content[:num_frame]
    else:
        original = original[:num_frame]
        content = content[:num_frame]
    _, C, H, W = original[0].shape

    if pose:
        h_ps = [frame.unsqueeze(2).unsqueeze(3) for frame in original_p]  # T of (1, pose_dim, 1, 1)
    else:
        h_ps = [netEP(frame) for frame in original]  # T of (1, pose_dim, 1, 1)

    if repeat_cont:
        h_cs = [netEC(content[0]) for _ in content]
    else:
        h_cs = [netEC(frame) for frame in content]
    recs = [netD([h_c, Variable(h_p)]) for h_c, h_p in zip(h_cs, h_ps)]  # T of (1, C, H, W)

    to_plot = [[torch.zeros(C, W, H)] + [frame[0].data for frame in original],
               [content[0][0].data] + [rec[0 ].clone().data for rec in recs]]
    img = tensor_seq_to_img(to_plot)
    return img






