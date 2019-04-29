import torch
from torch import nn
from tqdm import tqdm

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


