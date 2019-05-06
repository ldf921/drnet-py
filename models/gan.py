from collections import OrderedDict

import torch
from torch import nn

import utils
from .discriminator import Discriminator

class DRGAN:
    def __init__(self, opt):
        self.opt = opt
        self.netEC, _, self.netD, _ = utils.get_initialized_network(opt)
        self.netR = Discriminator(3, 64)

        self._modules = ['netEC', 'netD', 'netR']
        self.build_optimizer()

    def named_modules(self):
        return [ (name, getattr(self, name)) for name in self._modules]

    def modules(self):
        return map(lambda name : getattr(self, name), self._modules)

    def __iter__(self):
        return self.modules()

    def build_optimizer(self):
        for name, module in self.named_modules():
            optim_name = name.replace('net', 'optimizer')
            setattr(self, optim_name, utils.get_optimizer(self.opt, module))

    def save(self, cp_path):
        torch.save(dict(self.named_modules()), cp_path)

    def train(self, criterions, x):
        opt = self.opt
        mse_criterion, bce_criterion = criterions

        x, p = x

        b = x[0].size(0)
        x = [t.split(b // 2, dim = 0) for t in x]
        p = [t.split(b // 2, dim = 0) for t in p]

        x_c1 = x[0][0]
        x_c2 = x[1][0]
        x_p1 = x[2][0]
        x_p2 = x[3][0]

        h_p1 = p[2][0]
        h_p2 = p[3][0]
        h_p1 = h_p1.unsqueeze(2).unsqueeze(3)

        h_c1 = self.netEC(x_c1)
        with torch.no_grad():
            h_c2f = self.netEC(x_c2)
            h_c2 = h_c2f[0] if opt.content_model[-4:] == 'unet' else h_c2f  # used as target for sim loss

        # similarity loss: ||h_c1 - h_c2||
        sim_loss = mse_criterion(h_c1[0] if opt.content_model[-4:] == 'unet' else h_c1, h_c2)

        # reconstruction loss: ||D(h_c1, h_p1), x_p1||
        rec = self.netD([h_c1, h_p1])
        rec_loss = mse_criterion(rec, x_p1)

        # scene discriminator loss: maximize entropy of output
        # target = torch.cuda.FloatTensor(opt.batch_size, 1).fill_(0.5)
        # out = self.netC([h_p1, h_p2])
        # sd_loss = bce_criterion(out, Variable(target))

        # full loss
        # loss = sim_loss + rec_loss + opt.sd_weight*sd_loss
        loss = sim_loss + rec_loss

        ret = OrderedDict()
        ret['sim_loss'] = sim_loss
        ret['rec_loss'] = rec_loss

        if opt.swap_loss == 'gan':
            ''' reconstruct using another pose code
            '''
            x_c2 = x[0][1]
            h_p2 = p[1][1]
            h_p2 = h_p2.flip(dims=[0]).unsqueeze(2).unsqueeze(3)

            h_c2 = self.netEC(x_c2)
            swap_rec = self.netD([h_c2, h_p2])

            gan_loss = self.gan_loss(self.netR(swap_rec), real=True)
            loss = loss + opt.sd_weight * gan_loss

            self.optimizerEC.zero_grad()
            self.optimizerD.zero_grad()
            loss.backward()
            self.optimizerEC.step()
            self.optimizerD.step()

            dis_loss = (self.gan_loss(self.netR(swap_rec.detach()), real=False) +
                        self.gan_loss(self.netR(x[2][1]), real=True)) / 2
            self.optimizerR.zero_grad()
            dis_loss.backward()
            self.optimizerR.step()

            ret['gan_loss'] = gan_loss
            ret['dis_loss'] = dis_loss
        else:
            raise NotImplementedError

        return ret

    def gan_loss(self, preds, real=True):
        if real:
            return nn.BCEWithLogitsLoss()(preds, torch.ones_like(preds))
        else:
            return nn.BCEWithLogitsLoss()(preds, torch.zeros_like(preds))



