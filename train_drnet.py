import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
import numpy as np
from torch.autograd import Variable
import utils
import itertools
from tqdm import tqdm
from typing import Tuple

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.5, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='logs', help='base directory to save logs')
parser.add_argument('--data_root', default='', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--content_dim', type=int, default=128, help='size of the content vector')
parser.add_argument('--pose_dim', type=int, default=10, help='size of the pose vector')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--dataset', default='moving_mnist', help='dataset to train with')
parser.add_argument('--max_step', type=int, default=20, help='maximum distance between frames')
parser.add_argument('--sd_weight', type=float, default=0.0001, help='weight on adversarial loss')
parser.add_argument('--sd_nf', type=int, default=100, help='number of layers')
parser.add_argument('--content_model', default='dcgan_unet', help='model type (dcgan | dcgan_unet | vgg_unet)')
parser.add_argument('--pose_model', default='dcgan', help='model type (dcgan | unet | resnet)')
parser.add_argument('--data_threads', type=int, default=5, help='number of parallel data loading threads')
parser.add_argument('--normalize', action='store_true', help='if true, normalize pose vector')
parser.add_argument('--data_type', default='drnet', help='speed up data loading for drnet training')
parser.add_argument('--pose', action='store_true', help='use the extracted pose code')


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
opt.log_dir = '%s/%s%dx%d/%s' % (opt.log_dir, opt.dataset, opt.image_width, opt.image_width, name)

os.makedirs('%s/rec/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/analogy/' % opt.log_dir, exist_ok=True)

print(opt)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


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

    netEC.apply(utils.init_weights)
    netEP.apply(utils.init_weights)
    netD.apply(utils.init_weights)
    netC.apply(utils.init_weights)

    return netEC, netEP, netD, netC


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


# --------- plotting functions ------------------------------------
def plot_rec(opt, models, x, epoch):
    netEC, netEP, netD, _ = models
    if opt.pose:
        x, p = x
        x_c = x[0]
        h_c = netEC(x_c)
        t = np.random.randint(1, opt.max_step)
        x_p = x[t]
        h_p = p[t]
        h_p = h_p.unsqueeze(2).unsqueeze(3)
    else:
        x_c = x[0]
        x_p = x[np.random.randint(1, opt.max_step)]

        h_c = netEC(x_c)
        h_p = netEP(x_p)

    rec = netD([h_c, h_p])
    x_c, x_p, rec = x_c.data, x_p.data, rec.data
    fname = '%s/rec/%d.png' % (opt.log_dir, epoch)
    to_plot = []
    row_sz = 5
    nplot = 20
    for i in range(0, nplot-row_sz, row_sz):
        row = [[xc, xp, xr] for xc, xp, xr in zip(x_c[i:i+row_sz], x_p[i:i+row_sz], rec[i:i+row_sz])]
        to_plot.append(list(itertools.chain(*row)))
    utils.save_tensors_image(fname, to_plot)


def plot_analogy(opt, models, x, epoch):
    netEC, netEP, netD, _ = models
    if opt.pose:
        x, p = x
    x_c = x[0]

    h_c = netEC(x_c)
    nrow = 10
    row_sz = opt.max_step
    to_plot = []
    row = [xi[0].data for xi in x]
    zeros = torch.zeros(opt.channels, opt.image_width, opt.image_width)
    to_plot.append([zeros] + row)
    for i in range(nrow):
        to_plot.append([x[0][i].data])

    for j in range(0, row_sz):
        # h_p = netEP(x[j]).data
        h_p = p[j].unsqueeze(2).unsqueeze(3)
        for i in range(nrow):
            h_p[i] = h_p[0]
        rec = netD([h_c, Variable(h_p)])
        for i in range(nrow):
            to_plot[i+1].append(rec[i].data.clone())

    fname = '%s/analogy/%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)


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
    h_c2 = netEC(x_c2)[0].detach() if opt.content_model[-4:] == 'unet' else netEC(x_c2).detach() # used as target for sim loss

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

    return sim_loss.data.cpu().numpy(), rec_loss.data.cpu().numpy()


def train(models, optimizers, criterions, x):
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
    netEC, netEP, netD, netC = get_initialized_network(opt)
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

        for batch_idx, x in tqdm(enumerate(train_loader)):
            # train scene discriminator
            if opt.pose:
                sim_loss, rec_loss = train_pose(models, optimizers, criterions, x)
                epoch_sim_loss += sim_loss
                epoch_rec_loss += rec_loss
            else:
                sd_loss, sd_acc = train_scene_discriminator(models, optimizers, criterions, x)
                epoch_sd_loss += sd_loss
                epoch_sd_acc += sd_acc

                # train main model
                sim_loss, rec_loss = train(models, optimizers, criterions, x)
                epoch_sim_loss += sim_loss
                epoch_rec_loss += rec_loss

        # ---- eval phase
        for model in models:
            model.eval()
        # plot some stuff
        x = next(test_loader)
        plot_rec(opt, models, x, epoch)
        plot_analogy(opt, models, x, epoch)

        epoch_rec_loss = epoch_rec_loss/len(train_loader)
        epoch_sim_loss = epoch_sim_loss/len(train_loader)
        epoch_sd_acc = 100 * epoch_sd_acc/len(train_loader)
        ttl_samples = epoch * len(train_loader) * opt.batch_size
        print(f"{epoch: 02d} rec loss:{epoch_rec_loss:.4f} | sim loss: {epoch_sim_loss:.4f} | "
              f"scene disc acc: {epoch_sd_acc:.3f}% ({ttl_samples})")

        # save the model
        torch.save(
            {'netD': netD, 'netEP': netEP, 'netEC': netEC, 'opt': opt},
            f"{opt.log_dir}/model.pth"
        )


if __name__ == "__main__":
    main()
