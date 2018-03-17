import torch.nn as nn


def conv2d(in_c, out_c, k, s, p):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


def deconv2d(in_c, out_c, k, s, p, out_p):
    return nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, k, s, p, out_p),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True)
    )


def conv3d(in_c, out_c, k, s, p):
    return nn.Sequential(
        nn.Conv3d(in_c, out_c, k, s, p),
        nn.BatchNorm3d(out_c),
        nn.ReLU(True)
    )


def deconv3d(in_c, out_c, k, s, p, out_p):
    return nn.Sequential(
        nn.ConvTranspose3d(in_c, out_c, k, s, p, out_p),
        nn.BatchNorm3d(out_c),
        nn.ReLU
    )

class AudioDerivativeAutoEncoder(nn.Module):
    def __init__(self, derivative=True):
        super(AudioDerivativeAutoEncoder, self).__init__()
        self.derivative = derivative
        self.encoder = nn.Sequential(
            conv2d(1, 32, (3, 7), (1, 2), (1, 3)), # 128
            conv2d(32, 64, 3, 1, 1),
            conv2d(64, 128, 3, (1, 2), 1), # 64
            conv2d(128, 256, 3, 1, 1),
            conv2d(256, 512, 3, 2, 1), # 32
            conv2d(512, 512, 3, 1, 1),
            conv2d(512, 512, 3, 2, 1), # 16
        )

        self.decoder = nn.Sequential(
            deconv2d(512, 512, 3, 1, 1, 0),
            deconv2d(512, 256, 3, (1, 2), 1, (0, 1)),  # 32
            deconv2d(256, 128, 3, 1, 1, 0),
            deconv2d(128, 64, 3, (1, 2), 1, (0, 1)),  # 64
            deconv2d(64, 32, 3, 1, 1, 0),
            deconv2d(32, 32, 3, (1, 2), 1, (0, 1)), # 128
            nn.ConvTranspose2d(32, 1, 3, (1, 2), 1, (0, 1)) # 256
        )

    def forward(self, audio):
        # audio size: bchw = (b, 1, 16, 256)

        if not self.derivative:
            x = audio
        else:
            x = audio[:, :, 1:] - audio[:, :, :-1]
        x = self.encoder(x)
        return self.decoder(x)


class FlowAutoEncoder(nn.Module):
    def __init__(self, use_flow=True):
        super(FlowAutoEncoder, self).__init__()
        self.use_flow = use_flow

        in_c = 2 if use_flow else 3
        self.encoder = nn.Sequential(
            conv3d(in_c, 32, (3, 7, 7), (1, 2, 2), (1, 3, 3)),  # 32*32
            conv3d(32, 64, 3, 1, 1),
            conv3d(64, 128, 3, (1, 2, 2), 1), # 16*16
            conv3d(128, 256, 3, 1, 1),
            conv3d(256, 512, 3, (1, 2, 2), 1), # 8*8
            conv3d(512, 512, 3, 1, 1),
            conv3d(512, 512, 3, (1, 2, 2), 1)  # 4*4
        )

        self.decoder = nn.Sequential(
            deconv3d(512, 512, 3, 1, 1, 0),
            deconv3d(512, 512, 3, (1, 2, 2), 1, (0, 1, 1)), # 8*8
            deconv3d(512, 256, 3, 1, 1, 0),
            deconv3d(256, 128, 3, (1, 2, 2), 1, (0, 1, 1)), # 16*16
            deconv3d(128, 64, 3, 1, 1, 0),
            deconv3d(64, 32, 3, (1, 2, 2), 1, (0, 1, 1)), # 32*32
            nn.ConvTranspose3d(32, in_c, 3, (1, 2, 2), 1, (0, 1, 1)) # 64*64
        )

    def forward(self, input):
        # input size:
        # if imgs: bcthw = (b, 3, 16, 64, 64)
        # if flows = bcthw = (b, 2, 15, 64, 64)
        if not self.use_flow:
            x = input
        else:
            x = input[:, :, 1:] - input[:, :, :-1]

        x = self.encoder(x)
        return self.decoder(x)

