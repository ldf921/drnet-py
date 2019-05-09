import torch
from torch import nn
from torch.nn import functional as F


def conv_bn_relu(in_planes, out_planes, kernel_size, stride, padding):
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.2)
            )


def l2loss(x, y, squared=True, eps=1e-8):
    if squared:
        return (x - y).pow(2).sum(dim=1)
    else:
        return torch.sqrt((x - y).pow(2).sum(dim=1) + eps)


class ContentEncoder(nn.Module):
    def __init__(self, in_planes=3, normalize=False):
        super().__init__()

        modules = [
            conv_bn_relu(in_planes, 64, 4, 2, 1),
            conv_bn_relu(64, 128, 4, 2, 1),
            conv_bn_relu(128, 256, 4, 2, 1),
            conv_bn_relu(256, 512, 4, 2, 1),
            conv_bn_relu(512, 512, 4, 2, 1),
            nn.AdaptiveAvgPool2d((1, 1))
            ]
        self.module = nn.Sequential(*modules)
        self.normalize = normalize
        self.eps = 1e-8

    def forward(self, x):
        out = self.module(x).view(x.size(0), 512)
        if self.normalize:
            out = out / torch.sqrt((out ** 2).sum(dim=1, keepdim=True) + self.eps)
        return out


def triplet_loss(network, x_anchor, x_in, x_out, margin):
    """
    Triplet loss `x_acnhor` and `x_in` are two samples in the same video,
    `x_out` are generated frame
    Return
    ==============
    gan loss, discriminator loss
    """
    embedded_anchor = network(x_anchor)
    embedded_in = network(x_in)
    in_dist = l2loss(embedded_anchor, embedded_in)

    # triplet loss for discriminator, train embedding network, do not back prop to x_out
    dis_loss = F.relu(in_dist + margin - l2loss(embedded_anchor, network(x_out.detach())))

    # triplet loss for generator (maximize the loss), do not train embedding network. 
    gan_loss = -F.relu(in_dist.detach() + margin - l2loss(embedded_anchor.detach(), network(x_out)))

    return gan_loss.mean(), dis_loss.mean()