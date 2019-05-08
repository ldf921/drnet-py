from collections import OrderedDict
import torch
from torch import nn

from utils import utils
from .discriminator import Discriminator
from .base import Model


class DrGan(Model):
    def __init__(self, opt):
        assert opt.pose, "must use pose code"
        assert opt.swap_loss == "gan", "DrGan is supposed to use only for gan swap loss"
        self.opt = opt
        self.netEC, _, self.netD, _ = utils.get_initialized_network(opt)
        self.netR = Discriminator(layers=3, in_planes=3, first_out_planes=64)
        self._modules = ['netEC', 'netD', 'netR']

    def train_gan(self, criterions, x):
        opt = self.opt
        mse_criterion, bce_criterion = criterions
        ret = OrderedDict()

        x, p = x

        b = x[0].size(0)
        x = [t.split(b // 2, dim=0) for t in x]
        p = [t.split(b // 2, dim=0) for t in p]

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
        rec = self.netD([h_c1, h_p1])

        # similarity loss: ||h_c1 - h_c2||
        sim_loss = mse_criterion(h_c1[0] if opt.content_model[-4:] == 'unet' else h_c1, h_c2)
        ret['sim_loss'] = sim_loss.item()

        # reconstruction loss: ||D(h_c1, h_p1), x_p1||
        rec_loss = mse_criterion(rec, x_p1)
        ret['rec_loss'] = rec_loss.item()

        # sum of similarity loss and reconstruction loss
        loss = sim_loss + rec_loss

        # reconstruct using another pose code
        x_c2 = x[0][1]
        h_p2 = p[1][1]
        h_p2 = h_p2.flip(dims=[0]).unsqueeze(2).unsqueeze(3)
        h_c2 = self.netEC(x_c2)
        swap_rec = self.netD([h_c2, h_p2])

        # gan loss (generator part): - log(D(G(z))) as alternative to log(1 - D(G(z))) for stability
        gan_loss = self.gan_loss(self.netR(swap_rec), real=True)
        ret['gan_loss'] = gan_loss.item()
        loss = loss + opt.sd_weight * gan_loss
        self.optimizerEC.zero_grad()
        self.optimizerD.zero_grad()
        loss.backward()
        self.optimizerEC.step()
        self.optimizerD.step()

        # gan loss (discriminator part): - log(D(x)) - log(1 - D(G(z)))
        dis_loss = (self.gan_loss(self.netR(swap_rec.detach()), real=False) +
                    self.gan_loss(self.netR(x[2][1]), real=True)) / 2
        ret['dis_loss'] = dis_loss.item()
        self.optimizerR.zero_grad()
        dis_loss.backward()
        self.optimizerR.step()

        return ret

    def gan_loss(self, preds, real=True):
        if real:
            return nn.BCEWithLogitsLoss()(preds, torch.ones_like(preds))
        else:
            return nn.BCEWithLogitsLoss()(preds, torch.zeros_like(preds))



