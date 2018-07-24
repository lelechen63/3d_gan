import torch
import torch.nn as nn
from pts3d import *
from ops import *

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
            conv2d(128,128,3,2,1),
            Flatten(),
            linear(128*16*32,256),
                 
        )

        self.identy_conv1 = nn.Sequential(
            conv2d(3,96,7,2,1),
            )
        self.identy_conv2 = nn.Sequential(
            conv2d(96,256,5,2),
            conv2d(256,256,3,1,0),
            
            )
        self.identy_conv3 = nn.Sequential(
            nn.MaxPool2d(3,2),
            conv2d(256,512,3,1,1),
            conv2d(512,256,3,1,1),
            conv2d(256,128,3,1,1),
            Flatten(),
            linear(128*5*5,512),
            linear(512,256)
            )
        self.image_fc = nn.Sequential(
            linear(512,128))
        self.image_decoder1 = nn.Sequential(
            conv_transpose2d(128,512,7,2),
            conv_transpose2d(512, 256 ,6,2),
        )
        self.image_decoder2 = nn.Sequential(
            conv_transpose2d(512,256,6,2),
            conv_transpose2d(256,256,5,1),
            conv_transpose2d(256,96,5,1),
        )
        self.image_decoder3 = nn.Sequential(
            conv_transpose2d(96 * 2 , 96, 6, 2),
            conv_transpose2d(96,64,5,1),
            conv2d(64, 48, 3, 1, 1, activation=nn.Tanh, normalizer=None)
            )


    def forward(self,x,audio):
        b = self.identy_conv1(x)
        c= self.identy_conv2(b)
        d = self.identy_conv3(c)
        audio = self.cnn_extractor(audio)
        feature = torch.cat([audio,d],1)
        feature = self.image_fc(feature).view(feature.size(0),128,1,1)
        bb = self.image_decoder1(feature)
        bb =torch.cat([c,bb],1)
        cc = self.image_decoder2(bb)
        cc = torch.cat([b,cc],1)
        dd = self.image_decoder3(cc)
        dd =dd.view(dd.size(0),3,16,64,64)
        return dd



        