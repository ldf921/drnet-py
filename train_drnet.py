from collections import OrderedDict

import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
import utils
import itertools
# import progressbar
from shutil import copyfile
from tqdm import tqdm
from typing import Tuple

import valid
from metrics import Summary


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

opt = None


def get_optimizers(opt, models: Tuple[nn.Module]) -> Tuple[optim.Optimizer]:
    """
    :return: get a optimizer for each of the network with optimizer type, learning rate, and beta1 set as in `opt`
    """
    netEC, netEP, netD, netC = models
    if opt.optimizer == 'adam':
        opt.optimizer = optim.Adam
    elif opt.optimizer == 'rmsprop':
        opt.optimizer = optim.RMSprop
    elif opt.optimizer == 'sgd':
        opt.optimizer = optim.SGD
    else:
        raise ValueError('Unknown optimizer: %s' % opt.optimizer)

    optimizerEC = opt.optimizer(netEC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerEP = opt.optimizer(netEP.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerD = opt.optimizer(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerC = opt.optimizer(netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    return optimizerEC, optimizerEP, optimizerD, optimizerC


# --------- training functions ------------------------------------
def train_pose(models, optimizers, criterions, x):
    netEC, netEP, netD, _ = models
    optimizerEC, optimizerEP, optimizerD, _ = optimizers
    mse_criterion, bce_criterion = criterions

    x, p = x
    netEP.zero_grad()
    netEC.zero_grad()
    netD.zero_grad()

    x_c1 = x[0]
    x_c2 = x[1]
    x_p1 = x[2]
    x_p2 = x[3]

    h_p1 = p[2]
    h_p2 = p[3]
    h_p1 = h_p1.unsqueeze(2).unsqueeze(3)

    h_c1 = netEC(x_c1)
    with torch.no_grad():
        h_c2f = netEC(x_c2)
        h_c2 = h_c2f[0] if opt.content_model[-4:] == 'unet' else h_c2f  # used as target for sim loss

    # similarity loss: ||h_c1 - h_c2||
    sim_loss = mse_criterion(h_c1[0] if opt.content_model[-4:] == 'unet' else h_c1, h_c2)

    # reconstruction loss: ||D(h_c1, h_p1), x_p1||
    rec = netD([h_c1, h_p1])
    rec_loss = mse_criterion(rec, x_p1)

    # scene discriminator loss: maximize entropy of output
    # target = torch.cuda.FloatTensor(opt.batch_size, 1).fill_(0.5)
    # out = netC([h_p1, h_p2])
    # sd_loss = bce_criterion(out, Variable(target))

    # full loss
    # loss = sim_loss + rec_loss + opt.sd_weight*sd_loss
    loss = sim_loss + rec_loss
    loss.backward()
    optimizerEC.step()
    optimizerEP.step()
    optimizerD.step()

    ret = OrderedDict()
    ret['sim_loss'] = sim_loss
    ret['rec_loss'] = rec_loss

    if opt.swap_loss is not None:
        ''' reconstruct using another pose code
        '''
        with torch.no_grad():
            h_c2 = netEC(x_c2)
        h_p2 = h_p2.flip(dims=[0]).unsqueeze(2).unsqueeze(3)
        swap_rec = netD([h_c2, h_p2])
        h_c2_swap = netEC(swap_rec)
        swap_rec_loss = mse_criterion(h_c2_swap[0], h_c2[0])

        optimizerD.zero_grad()
        if opt.swap_loss == 'content':
            swap_rec_loss.backward()
            ret['swap_loss'] = swap_rec_loss
        else:
            raise NotImplementedError
        optimizerD.step()

    return ret


def train_encoder_decoder(models, optimizers, criterions, x):
    netEC, netEP, netD, netC = models
    optimizerEC, optimizerEP, optimizerD, optimizerC = optimizers
    mse_criterion, bce_criterion = criterions

    netEP.zero_grad()
    netEC.zero_grad()
    netD.zero_grad()

    x_c1 = x[0]
    x_c2 = x[1]
    x_p1 = x[2]
    x_p2 = x[3]

    h_c1 = netEC(x_c1)
    h_c2 = netEC(x_c2)[0].detach() if opt.content_model[-4:] == 'unet' else netEC(x_c2).detach()  # used as target for sim loss
    h_p1 = netEP(x_p1)  # used for scene discriminator
    h_p2 = netEP(x_p2).detach()

    # similarity loss: ||h_c1 - h_c2||
    sim_loss = mse_criterion(h_c1[0] if opt.content_model[-4:] == 'unet' else h_c1, h_c2)

    # reconstruction loss: ||D(h_c1, h_p1), x_p1||
    rec = netD([h_c1, h_p1])
    rec_loss = mse_criterion(rec, x_p1)

    # scene discriminator loss: maximize entropy of output
    target = torch.cuda.FloatTensor(opt.batch_size, 1).fill_(0.5)
    out = netC([h_p1, h_p2])
    sd_loss = bce_criterion(out, Variable(target))

    # full loss
    loss = sim_loss + rec_loss + opt.sd_weight*sd_loss
    loss.backward()

    optimizerEC.step()
    optimizerEP.step()
    optimizerD.step()

    return sim_loss.data.cpu().numpy(), rec_loss.data.cpu().numpy()


def train_scene_discriminator(models, optimizers, criterions, x):
    netEC, netEP, netD, netC = models
    optimizerEC, optimizerEP, optimizerD, optimizerC = optimizers
    mse_criterion, bce_criterion = criterions

    netC.zero_grad()
    target = torch.cuda.FloatTensor(opt.batch_size, 1)

    x1 = x[0]
    x2 = x[1]
    h_p1 = netEP(x1).detach()
    h_p2 = netEP(x2).detach()

    half = int(opt.batch_size/2)
    rp = torch.randperm(half).cuda()
    h_p2[:half] = h_p2[rp]
    target[:half] = 1
    target[half:] = 0

    out = netC([h_p1, h_p2])
    bce = bce_criterion(out, Variable(target))

    bce.backward()
    optimizerC.step()

    acc = out[:half].gt(0.5).sum() + out[half:].le(0.5).sum()
    return bce.data.cpu().numpy(), acc.data.cpu().numpy()/opt.batch_size

# --------- training loop ------------------------------------
def main():
    # load dataset
    train_loader, test_loader = utils.get_normalized_dataloader(opt)

    # get networks, criterions and optimizers
    netEC, netEP, netD, netC = utils.get_initialized_network(opt)
    models = (netEC, netEP, netD, netC)
    for model in models:
        model.cuda()

    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    criterions = (mse_criterion, bce_criterion)
    for criterion in criterions:
        criterion.cuda()

    # optimizer should be constructed after moving net to the device
    optimizerEC, optimizerEP, optimizerD, optimizerC = get_optimizers(opt, models)
    optimizers = (optimizerEC, optimizerEP, optimizerD, optimizerC)

    for epoch in range(opt.niter):
        # ---- train phase
        for model in models:
            model.train()

        epoch_sim_loss, epoch_rec_loss, epoch_sd_loss, epoch_sd_acc = 0, 0, 0, 0

        summary = Summary()
        for batch_idx, x in tqdm(enumerate(train_loader)):
            # train scene discriminator
            if opt.pose:
                summary.update(train_pose(models, optimizers, criterions, x))
            else:
                sd_loss, sd_acc = train_scene_discriminator(models, optimizers, criterions, x)
                epoch_sd_loss += sd_loss
                epoch_sd_acc += sd_acc

                # train main model
                sim_loss, rec_loss = train_encoder_decoder(models, optimizers, criterions, x)
                epoch_sim_loss += sim_loss
                epoch_rec_loss += rec_loss

        if opt.pose:
            print(f"[Epoch {epoch:03d}] {summary.format()}")
        else:
            epoch_rec_loss = epoch_rec_loss/len(train_loader)
            epoch_sim_loss = epoch_sim_loss/len(train_loader)
            epoch_sd_acc = 100 * epoch_sd_acc/len(train_loader)
            ttl_samples = epoch * len(train_loader) * opt.batch_size
            print(f"{epoch: 02d} rec loss:{epoch_rec_loss:.4f} | sim loss: {epoch_sim_loss:.4f} | "
                  f"scene disc acc: {epoch_sd_acc:.3f}% ({ttl_samples})")

        # ---- eval phase
        for model in models:
            model.eval()

        x = next(iter(test_loader))
        img = utils.plot_rec(opt.pose, models, x, opt.max_step)
        f_name = '%s/rec/%d.png' % (opt.log_dir, epoch)
        img.save(f_name)

        img = utils.plot_analogy(opt.pose, models, x, opt.channels, opt.image_width, opt.max_step)
        f_name = '%s/analogy/%d.png' % (opt.log_dir, epoch)
        img.save(f_name)

        # save the model
        torch.save(
            {'netD': netD, 'netEP': netEP, 'netEC': netEC, 'opt': opt},
            f"{opt.log_dir}/model.pth"
        )

        if epoch % 15 == 0:
            copyfile('%s/model.pth' % opt.log_dir, '%s/model-%d.pth' % (opt.log_dir, epoch))


def test():
    # load dataset
    train_loader, test_loader = utils.get_normalized_dataloader(opt)
    test_loader = iter(test_loader)

    cp = torch.load(os.path.join(opt.log_dir, opt.checkpoint))
    models = (cp['netEC'], cp['netEP'], cp['netD'], None)

    if opt.saveimg:
        valid.save_img(opt, models)
    else:
        rec_loss = valid.valid(opt, models, test_loader)
        print('rec_loss {:.6f}'.format(rec_loss))


if __name__ == "__main__":
    # load arguments
    opt = parser.parse_args()
    name = (f"content_model={opt.content_model}-"
            f"pose_model={opt.pose_model}-"
            f"content_dim={opt.content_dim}-"
            f"pose_dim={opt.pose_dim}-"
            f"max_step={opt.max_step}-"
            f"sd_weight={opt.sd_weight:.3f}-"
            f"lr={opt.lr:.3f}-"
            f"sd_nf={opt.sd_nf}-"
            f"normalize={opt.normalize}-"
            f"pose={int(opt.pose)}"
            )
    if opt.swap_loss is not None:
        name += f'-swap_loss={opt.swap_loss}'
    if len(opt.log_dir.split('/')) < 2:
        opt.log_dir = os.path.join(opt.log_dir, f"{opt.dataset}{opt.image_width}x{opt.image_width}", name)
    os.makedirs(os.path.join(opt.log_dir, "rec"), exist_ok=True)
    os.makedirs(os.path.join(opt.log_dir, "analogy"), exist_ok=True)

    # reset random seed
    print(opt)
    print("Log directory: {}".format(opt.log_dir))
    print("Random Seed: {}".format(opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    if opt.test:
        test()
    else:
        main()
