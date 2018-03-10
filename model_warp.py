import torch
import torch.nn as nn
from pts3d import *
from ops import *
import torchvision.models as models
import functools


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


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
            conv_block += [nn.ReflectionPad3d(1)]
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


class Generator(nn.Module):
    def __init__(self, batch_size, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='zero'):
        assert(n_blocks >= 0)
        super(Generator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        norm_layer = nn.BatchNorm2d

        self.audio_extractor = nn.Sequential(
            conv2d(1, 32, 3, 1, 1),
            conv2d(32, 64, 3, 1, 1),
            conv2d(64, 128, 3, (1, 2), 1),
            conv2d(128, 256, 3, 1, 1)  # 16,64
        )

        self.flow_predictor = nn.Sequential(
            conv3d(512, 64, 3, 1, 1),
            conv3d(64, 16, 3, 1, 1),
            conv3d(16, 4, 3, 1, 1),
            conv3d(4, 2, 3, 1, 1)
        )
        self.warp3d = Warp3D((batch_size, 2, 16, 8, 8))

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(3, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 3
        for i in range(n_downsampling):
            mult = 2**i
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

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
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
        image_feature = self.image_encoder(
            input).unsqueeze(2).repeat(1, 1, 16, 1, 1)
        audio_feature = self.audio_extractor(audio).view(-1, 256, 16, 8, 8)

        import pdb; pdb.set_trace()
        fused_feature = torch.cat([image_feature, audio_feature], 1)
        flows = self.flow_predictor(fused_feature)
        warped_feature = self.warp3d(image_feature, flows)

        out = self.compress(warped_feature)
        out = self.generator(out)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net_example = nn.Sequential(
            conv2d(3, 64, 4, 2, 1, normalizer=None),
            conv2d(64, 128, 4, 2, 1),
            conv2d(128, 256, 4, 2, 1),
            conv2d(256, 512, 4, 2, 1)
        )

        self.audio_extractor = nn.Sequential(
            conv2d(1, 32, 3, 1, 1, normalizer=None),
            conv2d(32, 64, 3, 2, 1, normalizer=None),  # 8,64
            conv2d(64, 128, 3, 1, 1, normalizer=None),
            conv2d(128, 256, 3, 2, 1, normalizer=None),  # 4,32
        )
        self.audio_fc = nn.Sequential(
            Flatten(),
            nn.Linear(256 * 4 * 32, 256),
            nn.ReLU(True)
        )

        self.net_image = nn.Sequential(
            conv3d(3, 64, 4, (2, 2, 2), 1, normalizer=None),
            conv3d(64, 128, 4, (2, 2, 2), 1),
            conv3d(128, 256, 4, (2, 2, 2), 1),
            conv3d(256, 512, 4, (1, 2, 2), 1)
        )
        #(512,1,4,4)

        self.net_joint = nn.Sequential(
            conv3d(512 + 256, 512, 3, 1, 1),
            conv3d(512, 1, (1, 4, 4), 1, 0,
                   activation=nn.Sigmoid, normalizer=None)
        )

    def forward(self, example_im, x, audio):
        v = self.net_image(x)
        audio = self.audio_extractor(audio)
        audio = self.audio_fc(audio)
        audio = audio.view(audio.size(0), audio.size(1), 1, 1, 1)
        audio = audio.repeat(1, 1, v.size(2), v.size(3), v.size(4))
        out = torch.cat([v, audio], 1)
        out = self.net_joint(out)
        score1 = out.view(out.size(0))

        v2 = self.net_image(x)
        example_im = self.net_example(example_im).unsqueeze(2)
        out = torch.cat([v2, example_im], 1)
        out = self.net_joint(out)
        score2 = out.view(out.size(0))

        return (score1 + score2) * 0.5


class Discriminator2(nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()

        self.audio_extractor = nn.Sequential(
            conv2d(1, 32, 3, 1, 1, normalizer=None),
            conv2d(32, 64, 3, 2, 1, normalizer=None),  # 16,64
            conv2d(64, 128, 3, 1, 1, normalizer=None),
            conv2d(128, 256, 3, 2, 1, normalizer=None),  # 16,32
        )
        self.audio_fc = nn.Sequential(
            Flatten(),
            nn.Linear(256 * 4 * 32, 256),
            nn.ReLU(True)
        )

        self.net_image = nn.Sequential(
            conv3d(3, 64, 4, (2, 2, 2), 1, normalizer=None),
            conv3d(64, 128, 4, (2, 2, 2), 1),
            conv3d(128, 256, 4, (2, 2, 2), 1),
            conv3d(256, 512, 4, (1, 2, 2), 1)
        )
        #(512,1,4,4)

        self.net_joint = nn.Sequential(
            conv3d(512 + 256, 512, 3, 1, 1),
            conv3d(512, 1, (1, 4, 4), 1, 0,
                   activation=nn.Sigmoid, normalizer=None)
        )

    def forward(self, example_im, x, audio):
        v = self.net_image(x)
        audio = self.audio_extractor(audio)
        audio = self.audio_fc(audio)
        audio = audio.view(audio.size(0), audio.size(1), 1, 1, 1)
        audio = audio.repeat(1, 1, v.size(2), v.size(3), v.size(4))
        out = torch.cat([v, audio], 1)
        out = self.net_joint(out)
        return out.view(out.size(0))


# a = torch.Tensor(1,3,16,64,64)
# a = Variable(a)
# net_image = nn.Sequential(
#             conv3d(3, 64, 4, (2,2,2), 1, normalizer=None),
#             conv3d(64, 128, 4, (2,2,2), 1),
#             conv3d(128, 256, 4, (2,2,2), 1),
#             conv3d(256, 512, 4, (1,2,2), 1)
#         )
# # b = torch.Tensor(1,1,64,128)
# # b = Variable(b).cuda()
# # netG = Generator().cuda()
# c = net_image(a)
# print c.size()
