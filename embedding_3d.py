import torch
import torch.nn as nn
from pts3d import *
from ops import *
import functools
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_nc = 3, output_nc = 3, ngf=64, norm_layer=nn.BatchNorm3d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(Encoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        model = [nn.ReplicationPad3d(3),
                 nn.Conv3d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv3d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=(1,2,2), padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.encoder = nn.Sequential(*model)
        model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose3d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=(1,2,2),
                                         padding=1, output_padding=(0,1,1),
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReplicationPad3d(3)]
        model += [nn.Conv3d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        # buf_f = []
        # buf_i = []
        # b,c,f,h,w = input.size()
        # for f_inx in range(f):
        #     buf_i.append(self.model(input[:,:,f_inx,:,:]).unsqueeze(2))
        #     buf_f.append(self.encoder(input[:,:,f_inx,:,:]).unsqueeze(2))
        # buf_f = torch.cat(buf_f,2)
        # buf_i = torch.cat(buf_i,2)

        f = self.encoder(input)
        # print f.size()
        i = self.model(f)
        # print i.size()
        return i, f


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad3d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out



def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# norm_layer = get_norm_layer(norm_type='instance')
# input_nc = 3
# output_nc = 3 
# ngf =64
# use_dropout = False
# netG = Encoder(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6).cuda()
# print netG
# a = torch.Tensor(1,3,16,64,64)
# a = Variable(a).cuda()
# b,c = netG(a)
# print b.size()
# print c.size()