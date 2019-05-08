import torch
from torch import nn

def conv_bn_relu(in_planes, out_planes, kernel_size, stride, padding):
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.2)
            )

class Discriminator(nn.Module):
    def __init__(self, layers, nf, in_planes=3):
        super().__init__()

        modules = [conv_bn_relu(in_planes, nf, 4, 2, 1)]
        for i in range(layers - 1):
            modules.append(conv_bn_relu(nf, nf * 2, 4, 2, 1))
            nf *= 2
        modules.append(nn.Conv2d(nf, 1, 4, stride=1, padding=0, bias=False))
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


