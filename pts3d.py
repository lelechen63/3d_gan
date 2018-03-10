import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class ResidualBlock3d(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResidualBlock3d, self).__init__()
        self.bottlenect = nn.Sequential(
            nn.Conv3d(channel_in, channel_out, 1),
            nn.BatchNorm3d(channel_out),
            nn.ReLU()
          )
        self.block = nn.Sequential(
            conv3d(channel_out, 4 * channel_out, 3, 1, 1),
            conv3d(4 * channel_out, channel_out, 3, 1, 1, activation=None)
        )

        self.lrelu = nn.ReLU(0.2)

    def forward(self, x):
        residual = self.bottlenect(x)
        out = self.block(residual)
        out += residual
        out = self.lrelu(out)
        return out


def linear(channel_in, channel_out,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm1d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Linear(channel_in, channel_out, bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def conv3d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm3d,
           groups=1):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv3d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=bias, groups=groups))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def conv_transpose3d(channel_in, channel_out,
                     ksize=4, stride=2, padding=1,
                     activation=nn.ReLU,
                     normalizer=nn.BatchNorm3d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.ConvTranspose3d(channel_in, channel_out,
                              ksize, stride, padding,
                              bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def nn_conv3d(channel_in, channel_out,
              ksize=3, stride=1, padding=1,
              scale_factor=2,
              activation=nn.ReLU,
              normalizer=nn.BatchNorm3d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Upsample(scale_factor=scale_factor))
    layer.append(nn.Conv3d(channel_in, channel_out,
                           ksize, stride, padding,
                           bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[1].weight)

    return nn.Sequential(*layer)


def _apply(layer, activation, normalizer, channel_out=None):
    if normalizer:
        layer.append(normalizer(channel_out))
    if activation:
        layer.append(activation())
    return layer

