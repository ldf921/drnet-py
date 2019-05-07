import torch
import torch.nn as nn
import argparse
import os
import random
from shutil import copyfile
from tqdm import tqdm

import valid
from utils import utils
from utils.metrics import Summary
from models.gan import DrGan
from models.drnet import DrNet


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


def main():
    # load dataset
    train_loader, test_loader = utils.get_normalized_dataloader(opt)

    # get networks and corresponding optimizers
    if opt.swap_loss == 'gan':
        models = DrGan(opt)
    else:
        models = DrNet(opt)
    models.cuda()
    models.build_optimizer()  # optimizers should be built after moving models to gpu

    # get criterions
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    criterions = (mse_criterion, bce_criterion)
    for criterion in criterions:
        criterion.cuda()

    # optimizer should be constructed after moving net to the device
    with open(os.path.join(opt.log_dir, 'log'), 'a') as fo:
        for epoch in range(opt.niter):
            # ---- train phase
            models.train()

            summary = Summary()
            for batch_idx, x in tqdm(enumerate(train_loader)):
                if opt.swap_loss == "gan" and opt.pose:
                    summary.update(models.train_gan(criterions, x))
                elif opt.pose:
                    summary.update(models.train_pose(criterions, x))
                else:
                    summary.update(models.train_scene_discriminator(criterions, x))
                    summary.update(models.train_encoder_decoder(criterions, x))

            utils.print_write_log(f"[Epoch {epoch:03d}] {summary.format()}", fo)

            # ---- eval phase
            models.eval()

            x = next(iter(test_loader))
            img = utils.plot_rec(opt.pose, models, x, opt.max_step)
            f_name = '%s/rec/%d.png' % (opt.log_dir, epoch)
            img.save(f_name)

            img = utils.plot_analogy(opt.pose, models, x, opt.channels, opt.image_width, opt.max_step)
            f_name = '%s/analogy/%d.png' % (opt.log_dir, epoch)
            img.save(f_name)

            # save the model
            cp_path = f"{opt.log_dir}/model.pth"
            models.save(cp_path)
            if epoch % 15 == 0:
                copyfile(
                    os.path.join(opt.log_dir, "model.pth"),
                    os.path.join(opt.log_dir, f"model-{epoch}.pth")
                )


def test():
    # load dataset
    train_loader, test_loader = utils.get_normalized_dataloader(opt)

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
            f"pose={int(opt.pose)}-"
            f"swap_loss={opt.swap_loss}"
            )
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
