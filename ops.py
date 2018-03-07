import torch
import torchvision
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from __future__ import print_function
from pts3d import conv3d


class ResidualBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            conv3d(channel_in, channel_out, 3, 1, 1),
            conv3d(channel_out, channel_out, 3, 1, 1, activation=None)
        )

        self.lrelu = nn.ReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.block(x)
        print(residual.size())
        print(x.size())
        print('-----')
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


def conv2d(channel_in, channel_out,
           ksize=3, stride=1, padding=1,
           activation=nn.ReLU,
           normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.Conv2d(channel_in, channel_out,
                     ksize, stride, padding,
                     bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def conv_transpose2d(channel_in, channel_out,
                     ksize=4, stride=2, padding=1,
                     activation=nn.ReLU,
                     normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.ConvTranspose2d(channel_in, channel_out,
                              ksize, stride, padding,
                              bias=bias))
    _apply(layer, activation, normalizer, channel_out)
    init.kaiming_normal(layer[0].weight)

    return nn.Sequential(*layer)


def nn_conv2d(channel_in, channel_out,
              ksize=3, stride=1, padding=1,
              scale_factor=2,
              activation=nn.ReLU,
              normalizer=nn.BatchNorm2d):
    layer = list()
    bias = True if not normalizer else False

    layer.append(nn.UpsamplingNearest2d(scale_factor=scale_factor))
    layer.append(nn.Conv2d(channel_in, channel_out,
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


class Warp(nn.Module):
    def __init__(self, flow_size):
        """Implementation of warping module including two steps:
            1. convert flow to sampling grid
            2. call grid_sample in PyTorch
        
        Arguments:
            nn {[type]} -- [description]
            flow_size {[type]} -- [description]
        """

        super(Warp, self).__init__()
        self.flow_size = flow_size
        B, C, H, W = flow_size
        assert C == 2
        h_coordinate = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1)
        w_coordinate = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W)

        grid_h = h_coordinate.expand(B, 1, H, W)
        grid_w = w_coordinate.expand(B, 1, H, W)

        self.grid_coordinate = Variable(torch.cat([grid_w, grid_h], 1))
    
    def forward(self, input, flow):
        assert flow.size() == self.flow_size
        flow[:, 0, :, :] = flow[:, 0, :, :] / float(self.flow_size[2]) * 2
        flow[:, 1, :, :] = flow[:, 1, :, :] / float(self.flow_size[3]) * 2
        grid = self.grid_coordinate + flow
        grid = grid.permute(0, 2, 3, 1)  # NCHW ==> NHWC
        return F.grid_sample(input, grid, padding_mode='border')
