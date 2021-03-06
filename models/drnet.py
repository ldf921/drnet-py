import torch
from torch.autograd import Variable
from collections import OrderedDict
from torch import nn

from utils import utils
from .base import Model


class DrNet(Model):
    def __init__(self, opt):
        assert opt.swap_loss is None, "swap loss must be none for DrNet"
        super().__init__(opt)
        self.opt = opt
        if self.opt.pose:
            self.netEC, _, self.netD, _ = utils.get_initialized_network(self.opt)
            self._modules = ["netEC", "netD"]
        else:
            self.netEC, self.netEP, self.netD, self.netC = utils.get_initialized_network(self.opt)
            self._modules = ["netEC", "netEP", "netD", "netC"]
        self.mse_criterion, self.bce_criterion = nn.MSELoss(), nn.BCELoss()
        self._criterions = ["mse_criterion", "bce_criterion"]

    def train(self, x):
        if self.opt.pose:
            return self.train_pose(x)
        else:
            return self.train_scene_discriminator(x) + self.train_encoder_decoder(x)

    def train_pose(self, x):
        """
        training assuming the pose code has been pre-extracted
        :return: sim_loss, rec_loss and swap_loss if specified in the option
        """
        opt = self.opt
        ret = OrderedDict()

        x, p = x

        # original frames
        x_0 = x[0]
        x_1 = x[1]
        x_2 = x[2]

        # extracted pose codes
        h_p2 = p[2]
        h_p3 = p[3]
        h_p2 = h_p2.unsqueeze(2).unsqueeze(3)

        # hidden content codes
        h_c0 = self.netEC(x_0)
        with torch.no_grad():
            h_c1f = self.netEC(x_1)
            h_c1 = h_c1f[0] if opt.content_model[-4:] == 'unet' else h_c1f  # used as target for sim loss

        # reconstruction using frame-0 hidden content code and frame-2 pose code
        rec = self.netD([h_c0, h_p2])

        # similarity loss: ||h_c0 - h_c1||
        sim_loss = self.mse_criterion(h_c0[0] if opt.content_model[-4:] == 'unet' else h_c0, h_c1)
        ret['sim_loss'] = sim_loss.item()

        # reconstruction loss: ||D(h_c0, h_p2), x_2||
        rec_loss = self.mse_criterion(rec, x_2)
        ret['rec_loss'] = rec_loss.item()

        # full loss = sim_loss + rec_loss
        loss = sim_loss + rec_loss

        self.optimizerEC.zero_grad()
        self.optimizerD.zero_grad()
        loss.backward()
        self.optimizerEC.step()
        self.optimizerD.step()

        if opt.swap_loss == "content":
            # reconstruct using another pose code
            with torch.no_grad():
                h_c1 = self.netEC(x_1)
            h_p3 = h_p3.flip(dims=[0]).unsqueeze(2).unsqueeze(3)
            swap_rec = self.netD([h_c1, h_p3])
            h_swap_c = self.netEC(swap_rec)
            swap_rec_loss = self.mse_criterion(h_swap_c[0], h_c1[0])
            ret['swap_loss'] = swap_rec_loss.item()

            self.optimizerD.zero_grad()
            swap_rec_loss.backward()
            self.optimizerD.step()
        return ret

    def train_encoder_decoder(self, x):
        """
        train the encoder-decoder cycle
        :return: `sim_loss` and `rec_loss`
        """
        ret = OrderedDict()

        x_c1 = x[0]
        x_c2 = x[1]
        x_p1 = x[2]
        x_p2 = x[3]

        h_c1 = self.netEC(x_c1)
        # used as target for sim loss
        h_c2 = self.netEC(x_c2)[0].detach() if self.opt.content_model[-4:] == 'unet' else self.netEC(x_c2).detach()
        # used for scene discriminator
        h_p1 = self.netEP(x_p1)
        h_p2 = self.netEP(x_p2).detach()
        rec = self.netD([h_c1, h_p1])

        # similarity loss: ||h_c1 - h_c2||
        sim_loss = self.mse_criterion(h_c1[0] if self.opt.content_model[-4:] == 'unet' else h_c1, h_c2)
        ret["sim_loss"] = sim_loss.item()

        # reconstruction loss: ||D(h_c1, h_p1), x_p1||
        rec_loss = self.mse_criterion(rec, x_p1)
        ret["rec_loss"] = rec_loss.item()

        # scene discriminator loss: maximize entropy of output
        target = torch.cuda.FloatTensor(self.opt.batch_size, 1).fill_(0.5)
        out = self.netC([h_p1, h_p2])
        sd_loss = self.bce_criterion(out, Variable(target))
        loss = sim_loss + rec_loss + self.opt.sd_weight*sd_loss

        # full loss
        self.optimizerEC.zero_grad()
        self.optimizerEP.zero_grad()
        self.optimizerD.zero_grad()
        loss.backward()
        self.optimizerEC.step()
        self.optimizerEP.step()
        self.optimizerD.step()

        return ret

    def train_scene_discriminator(self, criterions, x):
        """
        train the scene discriminator
        :return: `bce` and `acc`
        """
        mse_criterion, bce_criterion = criterions
        ret = OrderedDict()

        target = torch.cuda.FloatTensor(self.opt.batch_size, 1)
        x1 = x[0]
        x2 = x[1]
        h_p1 = self.netEP(x1).detach()
        h_p2 = self.netEP(x2).detach()

        half = int(self.opt.batch_size/2)
        rp = torch.randperm(half).cuda()
        h_p2[:half] = h_p2[rp]
        target[:half] = 1
        target[half:] = 0

        out = self.netC([h_p1, h_p2])
        bce = bce_criterion(out, Variable(target))
        ret["bce"] = bce.item()

        self.optimizerC.zero_grad()
        bce.backward()
        self.optimizerC.step()

        acc = out[:half].gt(0.5).sum() + out[half:].le(0.5).sum()
        ret["acc"] = 100 * acc.data.cpu().numpy()/self.opt.batch_size
        return ret


