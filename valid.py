import os

import torch
from torch import nn
from tqdm import tqdm

import utils

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def repeat_feature(tensor, T):
    if isinstance(tensor, tuple):
        return tuple([repeat_feature(t, T) for t in tensor])
    elif isinstance(tensor, list):
        return [repeat_feature(t, T) for t in tensor]
    else:
        return tensor.repeat((T, 1, 1, 1))


def valid_batch(opt, models, x):
    '''
        x : [T, B, C, H, W] list of tensor
        p : [T, B, 35] list of tensor
    '''
    netEC, netEP, netD, _ = models
    if opt.pose:
        x, p = x
    T = len(x)
    B = x[0].size(0)

    with torch.no_grad():
        h_c = repeat_feature(netEC(x[0]), T)               # repeat the content of the first frame
        if opt.pose:
            h_p = torch.cat(p, dim=0).unsqueeze(2).unsqueeze(3)
        else:
            h_p = netEP(torch.cat(x, dim=0))

        # reconstruction loss: ||D(h_c1, h_p1), x_p1||
        rec = netD([h_c, h_p])
        rec_loss = nn.MSELoss()(rec, torch.cat(x, dim=0))

        return rec_loss


def valid(opt, models, test_loader):
    rec_loss_avg = AverageMeter()
    for i, x in tqdm(zip(range(1000), test_loader)):
        rec_loss = valid_batch(opt, models, x)
        rec_loss_avg.update(rec_loss.item(), 1)
    return rec_loss_avg.avg


def tobatch(x):
    if isinstance(x, tuple):
        x, p = x
        return [x.unsqueeze(0), p.unsqueeze(0)]
    else:
        return x.unsqueeze(0)


def get_data(opt, data, indices=None):
    if indices is None:
        indices = data.get_index()
    x, flist = data.get_sequence_idx(*indices)
    x = tobatch(x)
    x = utils.normalize_data(opt, torch.cuda.FloatTensor, x)
    name = '_'.join(list(map(str, indices)))
    return x, name, flist

def save_img(opt, models):
    train_data, test_data = utils.load_data(opt)
    img_root = os.path.join(opt.log_dir, "swap")
    os.makedirs(img_root, exist_ok=True)
    for i in range(8):
        x_p, name_p, flist = get_data(opt, test_data)
        vid_root = os.path.join(img_root, name_p)
        os.makedirs(vid_root, exist_ok=True)
        with open(os.path.join(vid_root, 'flist.txt'), 'w') as fo:
            for path in flist:
                fo.write(path + '\n')
        for j in range(5):
            x_c, name_c, _ = get_data(opt, test_data)
            img = utils.plot_reconstr(opt.pose, models, x_p, x_c)
            img.save(os.path.join(vid_root, name_c + '.png'))

