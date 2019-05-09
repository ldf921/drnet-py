import math
from collections import OrderedDict

import torch
from torch import nn

from utils import utils
from .discriminator import Discriminator
from .base import Model
from .triplet import ContentEncoder, triplet_loss


def to_heatmap(pose, sigma=2):
    b = pose.size(0)
    size = 128
    p = 17
    identity = torch.Tensor([[[1, 0, 0], [0, 1, 0]]]).float().cuda()
    grid = torch.affine_grid_generator(identity.repeat(1, 1, 1), size=[1, 1, size, size])
    grid = grid.repeat(b, p, 1, 1, 1)
    partition = 2 * math.pi * sigma ** 2
    sigma = sigma / ((size - 1) / 2)
    pose = pose.narrow(1, 1, 34).view(b, p, 2)
    hm = torch.exp(-torch.sum((grid - pose.unsqueeze(2).unsqueeze(3)) ** 2, dim=4) / (2 * sigma ** 2)) / partition
    return hm


class CGan(Model):
    def __init__(self, opt):
        assert opt.pose, "must use pose code"
        assert opt.swap_loss == "cgan", "CGan is supposed to use only for cgan swap loss"
        super().__init__(opt)
        self.opt = opt
        self.netEC, _, self.netD, _ = utils.get_initialized_network(opt)
        self.netRP = Discriminator(layers=4, in_planes=17 + 3, first_out_planes=64)
        self.netRC = Discriminator(layers=4, in_planes=3 * 2, first_out_planes=64)

        self._modules = ['netEC', 'netD', 'netRP', 'netRC']

    def train(self, x):
        x, p = x

        b = x[0].size(0)
        x = [t.split(b // 2, dim=0) for t in x]
        p = [t.split(b // 2, dim=0) for t in p]

        x_c = x[0][0]
        h_p = p[1][0]

        h_c = self.netEC(x_c)
        x_fake = self.netD([h_c, h_p.unsqueeze(2).unsqueeze(3)])

        h_p = to_heatmap(h_p)
        pose_loss = self.gan_loss(self.netRP(torch.cat([x_fake, h_p], dim=1)), real=True)
        content_loss = self.gan_loss(self.netRC(torch.cat([x_fake, x_c], dim=1)), real=True)
        loss = pose_loss + content_loss

        self.optimizerEC.zero_grad()
        self.optimizerD.zero_grad()
        loss.backward()
        self.optimizerEC.step()
        self.optimizerD.step()

        dis_pose = (self.gan_loss(self.netRP(torch.cat([x_fake.detach(), h_p], dim=1)), real=False) +
                    self.gan_loss(self.netRP(torch.cat([x[0][1], to_heatmap(p[0][1])], dim=1)), real=True)) / 2
        self.optimizerRP.zero_grad()
        dis_pose.backward()
        self.optimizerRP.step()

        dis_content = (self.gan_loss(self.netRC(torch.cat([x_fake.detach(), x_c], dim=1)), real=False) +
                       self.gan_loss(self.netRC(torch.cat([x[0][1], x[1][1]], dim=1)), real=True)) / 2
        self.optimizerRC.zero_grad()
        dis_content.backward()
        self.optimizerRC.step()

        ret = OrderedDict()
        ret['gan_pose'] = pose_loss
        ret['gan_content'] = content_loss
        ret['dis_pose'] = dis_pose
        ret['dis_content'] = dis_content

        return ret

    def gan_loss(self, preds, real=True):
        if real:
            return nn.BCEWithLogitsLoss()(preds, torch.ones_like(preds))
        else:
            return nn.BCEWithLogitsLoss()(preds, torch.zeros_like(preds))


class CGanTriplet(Model):
    def __init__(self, opt):
        assert opt.pose, "must use pose code"
        assert opt.swap_loss == "cgan-triplet", "CGanTriplet is supposed to use only for `cgan-triplet` swap loss"
        super().__init__(opt)
        self.opt = opt
        self.netEC, _, self.netD, _ = utils.get_initialized_network(opt)
        self.netRP = Discriminator(layers=4, in_planes=17 + 3, first_out_planes=64)
        self.netRC = ContentEncoder()

        self._modules = ['netEC', 'netD', 'netRP', 'netRC']
        self.margin = self.opt.cgan_triplet["margin"]

    def train(self, x):
        x, p = x

        b = x[0].size(0)
        x = [t.split(b // 2, dim=0) for t in x]
        p = [t.split(b // 2, dim=0) for t in p]

        x_c = x[0][0]
        h_p = p[1][0]

        h_c = self.netEC(x_c)
        x_fake = self.netD([h_c, h_p.unsqueeze(2).unsqueeze(3)])

        h_p = to_heatmap(h_p)
        pose_loss = self.gan_loss(self.netRP(torch.cat([x_fake, h_p], dim=1)), real=True)
        content_loss, dis_content = triplet_loss(self.netRC, x_c, x[2][0], x_fake, margin=self.margin)
        loss = pose_loss + content_loss

        # optimizing generator (Encoder - Decoder)
        self.optimizerEC.zero_grad()
        self.optimizerD.zero_grad()
        loss.backward()
        self.optimizerEC.step()
        self.optimizerD.step()

        # optimizing pose discriminator
        dis_pose = (self.gan_loss(self.netRP(torch.cat([x_fake.detach(), h_p], dim=1)), real=False) +
                    self.gan_loss(self.netRP(torch.cat([x[0][1], to_heatmap(p[0][1])], dim=1)), real=True)) / 2
        self.optimizerRP.zero_grad()
        dis_pose.backward()
        self.optimizerRP.step()

        # optimizing content discriminator
        self.optimizerRC.zero_grad()
        dis_content.backward()
        self.optimizerRC.step()

        ret = OrderedDict()
        ret['gan_pose'] = pose_loss
        ret['gan_content'] = content_loss
        ret['dis_pose'] = dis_pose
        ret['dis_content'] = dis_content

        return ret

    def gan_loss(self, preds, real=True):
        if real:
            return nn.BCEWithLogitsLoss()(preds, torch.ones_like(preds))
        else:
            return nn.BCEWithLogitsLoss()(preds, torch.zeros_like(preds))
