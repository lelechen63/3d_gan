import torch
import torch.nn as nn
from pts3d import *
from ops import *

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

noise_size = 100

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.cnn_extractor = nn.Sequential(
            conv2d(1,32,3,1,1),
            conv2d(32,64,3,2,1),
            conv2d(64,128,3,1,1),
            conv2d(128,128,3,2,1)
        )

        self.audio_fc= nn.Sequential(
            Flatten(),
            linear(128*16*32,256),
        )

        self.static_net = nn.Sequential(
            conv_transpose2d(256 + 100,512,4,1,0),

            conv_transpose2d(512,256,4,2,1 ),

            conv_transpose2d(256,128,4,2,1 ),

            conv_transpose2d(128,64,4,2,1),

            conv_transpose2d(64,64,4,2,1),
            conv2d(64, 3, 3, 1, 1, activation=nn.Tanh, normalizer=None)
            )

        self.net_video = nn.Sequential(
            conv_transpose3d(100 + 256,512,(2,4,4),1,0 ),

            conv_transpose3d(512,256,(2,4,4),2,1),

            conv_transpose3d(256,128,4,2,1),

            conv_transpose3d(128,64,4,2,1 ),
 
            )

        self.mask_net = nn.Sequential(
            conv_transpose3d(64,64,4,2,1),
            conv3d(64, 1, 3, 1, 1, activation=nn.Sigmoid, normalizer=None)
            )
        self.gen_net = nn.Sequential(
            conv_transpose3d(64,64,4,2,1 ),
            conv3d(64, 3, 3, 1, 1, activation=nn.Tanh, normalizer=None)
            )


    def forward(self,x,z,audio):

        feature = self.cnn_extractor(audio)
        feature = self.audio_fc(feature)
        out  = torch.cat([z, feature], 1)

        z_2d = out.view(out.size(0),256 + 100,1,1)
        z_3d = out.view(out.size(0),256 + 100,1,1,1)
        temp = self.net_video(z_3d)
        f = self.gen_net(temp)
        m = self.mask_net(temp)
        b = self.static_net(z_2d)

        b = torch.unsqueeze(b,2)
        b = b.repeat(1,1,m.size(2),1,1)
        result = torch.add(torch.mul(f,m) , torch.mul(torch.add(torch.mul(m,-1),1),b))

        return result

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
            nn.Linear(256*16*32,256),
            nn.ReLU()
        )
        self.netD = nn.Sequential(
            conv3d(3,64,4,2,1, ),
            conv3d(64,128,4,2,1, ),
            conv3d(128,256,4,2,1, ),
            conv3d(256,512,(2,4,4),2,1, ),
        )
        self.net_joint = nn.Sequential(
            conv3d(512 + 256 ,1,(2,4,4),1,0,activation = nn.Sigmoid,normalizer = None),
            )
        
    def forward(self, x,audio):
        x = self.netD(x)
        audio = self.cnn_extractor(audio)
        audio = self.audio_fc(audio)
        audio = audio.view(audio.size(0), audio.size(1), 1, 1, 1)
        audio = audio.repeat(1, 1, x.size(2), x.size(3),x.size(4))

        out = torch.cat([x,audio],1)

        out = self.net_joint(out)
        out = out.view(out.size(0))
        return out
