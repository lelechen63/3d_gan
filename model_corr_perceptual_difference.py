import functools

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from ops import *
from pts3d import *


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Generator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9,
                 padding_type='zero'):
        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        norm_layer = nn.BatchNorm2d

        self.audio_extractor = nn.Sequential(
            conv2d(1, 32, 3, 1, 1),
            conv2d(32, 64, 3, 2, 1),
            conv2d(64, 128, 3, 1, 1),
            conv2d(128, 256, 3, 2, 1),
            conv2d(256, 256, 3, stride=(1, 2), padding=1)
            # nn.MaxPool2d((1,2),(1,2))
        )

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.image_encoder = nn.Sequential(*model)

        norm_layer = nn.BatchNorm3d
        self.compress = nn.Sequential(
            nn.Conv3d(ngf * 8, ngf * mult * 2, kernel_size=3,
                      stride=1, padding=1, bias=use_bias),
            norm_layer(ngf * mult * 2),
            nn.ReLU(True)

        )

        model = []

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=(3, 3, 3), stride=(1, 2, 2),
                                         padding=(1), output_padding=(0, 1, 1),
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.generator = nn.Sequential(*model)

    def forward(self, input, audio):
        image_feature = self.image_encoder(input).unsqueeze(
            2).repeat(1, 1, audio.size(2) / 4, 1, 1)
        audio_feature = self.audio_extractor(audio)
        repeated_audio_feature = audio_feature.unsqueeze(-1).repeat(1, 1, 1, 1, image_feature.size(-1))

        new_input = torch.cat([image_feature, repeated_audio_feature], 1)
        out = self.compress(new_input)
        out = self.generator(out)
        return out, audio_feature


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = (0, 1, 1)
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = (0, 1, 1)
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=(1, 3, 3), padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.cnn_extractor = nn.Sequential(
            conv2d(1,32,3,1,1),
            conv2d(32,64,3,2,1),
            conv2d(64,128,3,1,1),
            conv2d(128,256,3,2,1)
        )

        self.audio_fc= nn.Sequential(
            Flatten(),
            linear(256*16*32,256),
        )
        self.net_image = nn.Sequential(
            conv3d(3, 64, 4, (2,2,2), 1, normalizer=None),
            conv3d(64, 128, 4, (2,2,2), 1),
            conv3d(128, 256, 4, (2,2,2), 1),
            conv3d(256, 256, 4, (1,2,2), 1)
        )
        self.net_flow_diff = nn.Sequential(
            conv3d(2, 64, 3, 2, 1, normalizer=None),
            conv3d(64, 128, 3, (2, 1, 1), 1),
            conv3d(128, 256, 3, 2, 1),
            conv3d(256, 256, 3, (2, 1, 1), 1)
        )

        self.net_joint = nn.Sequential(
            conv3d(512 + 256, 512, 3, 1, 1),
            conv3d(512, 1, (1,4,4), 1, 0, activation=nn.Sigmoid, normalizer=None)
        )

    def forward(self, imgs, flows, audio):
        diff = self.net_flow_diff(flows)
        imgs = self.net_image(imgs)
        audio = self.cnn_extractor(audio)
        audio = self.audio_fc(audio)
        audio = audio.view(audio.size(0), audio.size(1), 1, 1, 1)
        audio = audio.repeat(1, 1, imgs.size(2), imgs.size(3), imgs.size(4))

        out = torch.cat([imgs, diff, audio], 1)
        out = self.net_joint(out)
        return out.view(out.size(0))


def gen_clip_derivative(real_im):
    assert isinstance(real_im, Variable)
    # real_im must in shape of (batch, channel, depth, height, width)
    assert len(real_im.size()) == 5

    # permute from BCDHW to BDHWC
    real_im = real_im.permute(0, 2, 3, 4, 1)
    np_imgs = real_im.cpu().data.numpy()

    b, d, h, w, c = real_im.size()
    of_size = (b, d - 1, h, w, 2)
    optical_flows = torch.FloatTensor(*of_size).zero_()
    optical_flows = Variable(optical_flows).cuda()
    for idx_b in range(b):
        for i in range(d - 1):
            prev_frm, next_frm = np_imgs[idx_b, i], np_imgs[idx_b, i + 1]
            prev_frm = cv2.cvtColor(prev_frm, cv2.COLOR_RGB2GRAY)
            next_frm = cv2.cvtColor(next_frm, cv2.COLOR_RGB2GRAY)
            of = cv2.calcOpticalFlowFarneback(
                prev_frm, next_frm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            of = torch.from_numpy(of)
            of = Variable(of)
            optical_flows[idx_b, i] = of
    # permute from BDHWC back to BCDHW
    optical_flows = optical_flows.permute(0, 4, 1, 2, 3)
    return optical_flows


class ClipFlowConv(nn.Module):
    def __init__(self, feature_len=1):
        super(ClipFlowConv, self).__init__()
        assert feature_len == 16

        self.feature_len = feature_len
        if feature_len == 16:
            self.convs = nn.Sequential(
                nn.Conv3d(2, 4, 3, (1, 2, 2), 1),
                nn.InstanceNorm3d(4),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(4, 8, 3, (1, 2, 2), 1),
                nn.InstanceNorm3d(8),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(8, 16, 3, (1, 2, 2), 1),
                nn.InstanceNorm3d(16),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                nn.InstanceNorm3d(16),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                nn.InstanceNorm3d(16),
                nn.LeakyReLU(inplace=True),
                nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                nn.InstanceNorm3d(16),
                nn.LeakyReLU(inplace=True)
            )
        elif feature_len == 1:
            self.conv = nn.Conv2d(2, 1, 64, 1, 0, bias=False)

    def forward(self, flows):
        out = self.convs(flows)
        # permute from BCDHW to BDCHW
        return out.permute(0, 2, 1, 3, 4).contiguous().view(-1, 15, 16)


def visualize_of(flow):
    assert isinstance(flow, np.ndarray)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    h, w, _ = flow.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


class AudioDerivative(nn.Module):
    def __init__(self, feature_len=16):
        super(AudioDerivative, self).__init__()
        assert feature_len == 16
        self.feature_len = feature_len
        self.convs = nn.Sequential(nn.Conv2d(128, 64, 3, (1, 2), 1),
                                   nn.InstanceNorm2d(64),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(64, 32, 3, (1, 2), 1),
                                   nn.InstanceNorm2d(32),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(32, 16, 3, (1, 2), 1),
                                   nn.InstanceNorm2d(16),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, (1, 2), 1),
                                   nn.InstanceNorm2d(16),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Conv2d(16, 16, 3, (1, 2), 1),
                                   nn.InstanceNorm2d(16),
                                   nn.LeakyReLU(inplace=True)
                                   )

    def forward(self, input):
        b, c, h, w = input.size()
        buf = []
        for idx_h in range(h - 1):
            diff = input[:, :, idx_h, :] - input[:, :, idx_h + 1, :]
            diff = torch.unsqueeze(diff, 2)
            buf.append(diff)
        buf = torch.cat(buf, 2)
        out = self.convs(buf)
        return out.permute(0, 2, 1, 3).contiguous().view(-1, h - 1, 16)


class PearsonCorrelationCoefficient(nn.Module):
    def __init__(self):
        super(PearsonCorrelationCoefficient, self).__init__()
        self.avg_pool = nn.AvgPool1d(16, stride=16)
        self.instance_norm = nn.InstanceNorm1d(15)

    def forward(self, f1, f2):
        assert f1.size() == f2.size()
        b, t, l = f1.size()

        f1 = self.avg_pool(f1)
        f1 = self.instance_norm(f1).view(-1, 15)
        f2 = self.avg_pool(f2)
        f2 = self.instance_norm(f1).view(-1, 15)

        return torch.sum(f1 * f2, 1) / 14
