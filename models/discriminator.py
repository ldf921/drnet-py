from torch import nn


def conv_bn_relu(in_planes, out_planes, kernel_size, stride, padding):
    return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.LeakyReLU(0.2)
            )


class Discriminator(nn.Module):
    def __init__(self, layers, in_planes, first_out_planes):
        """
        :param layers: number of `conv_bn_relu` blocks
        :param in_planes: input number of channels
        :param first_out_planes: first block number of channels, thus final output would have `out_planes * 2 ** (layers - 1)`
                           channels
        """
        super().__init__()
        modules = [conv_bn_relu(in_planes, first_out_planes, 4, 2, 1)]
        for i in range(layers - 1):
            modules.append(conv_bn_relu(first_out_planes, first_out_planes * 2, 4, 2, 1))
            first_out_planes *= 2
        modules.append(nn.Conv2d(first_out_planes, 1, 4, stride=1, padding=0, bias=False))
        self.module = nn.Sequential(*modules)

    def forward(self, x):
        return self.module(x)


