import torch
import torch.nn as nn
from pts3d import *
from ops import *
import torchvision.models as models
import functools


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.face_extractor1 = nn.Sequential(
            conv2d(3,128,3,1,1),
        )
        self.face_extractor2 = nn.Sequential(
            conv2d(128,256,3,2,1),
        )

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
        self.fc_layer = nn.Sequential(
            linear(256 ,1024*4*4)
            )
        self.net_joint1 = nn.Sequential(
            conv_transpose3d(1024, 1024, (1,1,1), 1,0),
            conv_transpose3d(1024, 512, 4, (2,2,2),1),
            conv_transpose3d(512, 256, 4, (2,2,2),1),
            conv_transpose3d(256, 256, 4,(2,2,2),1),
            )
        self.net_joint2 = nn.Sequential(
            conv_transpose3d(256 + 256, 128, 4,(2,2,2),1),
        )
        self.net_joint3 = nn.Sequential(
            conv3d(128 + 128 , 64 , 3 ,1,1),
            conv3d(64, 3, 3, 1, 1, activation=nn.Tanh, normalizer=None)
        )


    def forward(self, x, y,audio):
        face_f1 = self.face_extractor1(x)
        face_f2 = self.face_extractor2(face_f1)

        feature = self.cnn_extractor(audio)
        feature = self.audio_fc(feature)
        out = feature
        # out  = torch.cat([z, feature], 1)
        out = self.fc_layer(out)
        out = out.view(out.size()[0], 1024,1,4,4)
        out = self.net_joint1(out)
        face_f2 = face_f2.view(face_f2.size()[0],face_f2.size()[1],1,face_f2.size()[2],face_f2.size()[3])

        face_f2 = face_f2.repeat(1,1,out.size(2),1,1)
        out = torch.cat([face_f2,out],1)
        out = self.net_joint2(out)
        face_f1 = face_f1.view(face_f1.size(0),face_f1.size(1),1,face_f1.size(2),face_f1.size(3))
        face_f1 = face_f1.repeat(1,1,out.size(2),1,1)
        out = torch.cat([face_f1,out],1)
        out = self.net_joint3(out)

        return out



