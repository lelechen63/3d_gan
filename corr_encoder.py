import torch
import torch.nn as nn
import torch.nn.functional as F

from ops import *
from pts3d import *

__all__ = ['FlowEncoder', 'AudioDeriEncoder', '']

class FlowEncoder(nn.Module):
    def __init__(self, flow_type='farneback'):
        super(FlowEncoder, self).__init__()
        self.flow_type = flow_type
        if flow_type == 'farneback':
            self.convs = nn.Sequential(
                            nn.Conv3d(2, 4, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(4),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(4, 8, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(8),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(8, 16, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(16),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(16),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(16),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(16),
                            nn.ReLU(inplace=True)
                            )
        elif flow_type == 'flownet':
            self.convs = nn.Sequential(
                            nn.Conv3d(2, 4, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(4),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(4, 8, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(8),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(8, 16, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(16),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                            nn.BatchNorm3d(16),
                            nn.ReLU(inplace=True)
                            )
        elif flow_type == 'embedding':
            self.convs = nn.Sequential(
                nn.Conv3d(3, 4, 3, (1, 2, 2), 1),
                nn.BatchNorm3d(4),
                nn.ReLU(inplace=True),
                nn.Conv3d(4, 8, 3, (1, 2, 2), 1),
                nn.BatchNorm3d(8),
                nn.ReLU(inplace=True),
                nn.Conv3d(8, 16, 3, (1, 2, 2), 1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.Conv3d(16, 16, 3, (1, 2, 2), 1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True)
            )
        

    def forward(self, flows):
        out = self.convs(flows)
        # permute from BCDHW to BDCHW
        if self.flow_type != 'embedding':
            return out.permute(0, 2, 1, 3, 4).contiguous().view(-1, 15, 16)
        else:
            return out.permute(0, 2, 1, 3, 4).contiguous().view(-1, 16, 16)


class AudioDeriEncoder(nn.Module):
    def __init__(self, need_deri=True):
        super(AudioDeriEncoder, self).__init__()
        self.need_deri = need_deri
        self.convs = nn.Sequential(nn.Conv2d(256, 128, 3, (1, 2), 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, 3, (1, 2), 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 32, 3, (1, 2), 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(32, 16, 3, (1, 2), 1),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True)
                                   )

    def forward(self, input):
        b, c, h, w = input.size()
        if self.need_deri:
            buf = []
            for idx_h in range(h-1):
                diff = input[:, :, idx_h, :] - input[:, :, idx_h+1, :]
                diff = torch.unsqueeze(diff, 2)
                buf.append(diff)
            buf = torch.cat(buf, 2)
            out = self.convs(buf)
            return out.permute(0, 2, 1, 3).contiguous().view(-1, h-1, 16)
        else:
            out = self.convs(input)

            return out.permute(0, 2, 1, 3).contiguous().view(-1, h, 16)


class FlowResEncoder(nn.Module):
    def __init__(self, input_nc=2, ngf=256, use_dropout=False, n_blocks=9, flow_type='farneback'):
        assert(n_blocks >= 0)
        super(FlowResEncoder, self).__init__()
        assert flow_type == 'farneback' or flow_type == 'flownet'
        self.input_nc = input_nc
        self.ngf = ngf

        use_bias = False

        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 nn.BatchNorm3d(ngf),
                 nn.ReLU(True)]

        if flow_type == 'farneback':
            n_downsampling = 6
            out_channels = [256, 128, 64, 32, 16, 16, 16]
        else:
            n_downsampling = 4
            out_channels = [256, 128, 64, 32, 16]

        for i in range(n_downsampling):
            model += [nn.Conv3d(out_channels[i], out_channels[i+1], kernel_size=3,
                                stride=(1, 2, 2), padding=1, bias=use_bias),
                      nn.BatchNorm3d(out_channels[i+1]),
                      nn.ReLU(True)]

        for i in range(n_blocks):
            model += [ResnetBlock3d(16, use_dropout=use_dropout, use_bias=use_bias)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input).view(-1, 16, 15).permute(0, 2, 1)

# Define a 3d resnet block
class ResnetBlock3d(nn.Module):
    def __init__(self, dim, use_dropout, use_bias):
        super(ResnetBlock3d, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_dropout, use_bias)

    def build_conv_block(self, dim, use_dropout, use_bias):
        conv_block = []
        p = 0
        conv_block += [nn.ReplicationPad3d(1)]

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       nn.BatchNorm3d(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        conv_block += [nn.ReplicationPad3d(1)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       nn.BatchNorm3d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Define a 3d resnet block
class ResnetBlock2d(nn.Module):
    def __init__(self, dim, use_dropout, use_bias):
        super(ResnetBlock2d, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_dropout, use_bias)

    def build_conv_block(self, dim, use_dropout, use_bias):
        conv_block = []
        p = 0
        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       nn.BatchNorm2d(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       nn.BatchNorm2d(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class AudioDeriResEncoder(nn.Module):
    def __init__(self, input_nc=256, ngf=256, use_dropout=False, n_blocks=9):
        super(AudioDeriResEncoder, self).__init__()
        self.input_nc = input_nc
        self.ngf = ngf

        use_bias = False

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 nn.BatchNorm2d(ngf),
                 nn.ReLU(True)]

        n_downsampling = 4
        out_channels = [256, 128, 64, 32, 16]

        for i in range(n_downsampling):
            model += [nn.Conv2d(out_channels[i], out_channels[i+1], kernel_size=3,
                                stride=(1, 2), padding=1, bias=use_bias),
                      nn.BatchNorm2d(out_channels[i+1]),
                      nn.ReLU(True)]

        for i in range(n_blocks):
            model += [ResnetBlock2d(16, use_dropout=use_dropout, use_bias=use_bias)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        b, c, h, w = input.size()
        buf = []
        for idx_h in range(h-1):
            diff = input[:, :, idx_h, :] - input[:, :, idx_h+1, :]
            diff = torch.unsqueeze(diff, 2)
            buf.append(diff)
        buf = torch.cat(buf, 2)
        return self.model(buf).view(-1, 16, 15).permute(0, 2, 1)


class CosineSimiLoss(nn.Module):
    def __init__(self, batch_size):
        super(CosineSimiLoss, self).__init__()
        self.ones = Variable(torch.ones(batch_size)).cuda(async=True)
        self.l1_loss = nn.L1Loss()
    
    def forward(self, input1, input2):
        cos_simi = F.cosine_similarity(input1, input2, eps=1e-5)
        return self.l1_loss(cos_simi, self.ones)

# import torch
# net = torch.nn.DataParallel(AudioDeriEncoder().cuda())
# t = torch.cuda.FloatTensor(12, 256, 16, 16)
# t = torch.autograd.Variable(t)
# out = net(t)
# print(out.size())
