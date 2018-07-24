import os
import glob
import time
import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import VaganDataset
from embedding_3d import Encoder
from tensorboard_logger import configure, log_value

class Trainer():
    def __init__(self, config):
        self.encoder = Encoder()

        print(self.encoder)
        self.l1_loss_fn =  nn.MSELoss()
        self.mse_loss_fn = nn.MSELoss()
        self.opt_g = torch.optim.Adam(self.encoder.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))
        

        self.dataset = VaganDataset(config.dataset_dir, train=config.is_train)
    
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=4,
                                      shuffle=True, drop_last=True)
        data_iter = iter(self.data_loader)
        data_iter.next()

        if config.cuda:

           

            self.encoder     = self.encoder.cuda()
            self.l1_loss_fn = self.l1_loss_fn.cuda()

        self.config = config
        self.start_epoch = 0
        # self.load()
    def fit(self):
        config = self.config
        configure("{}/".format(config.log_dir), flush_secs=5)
        num_steps_per_epoch = len(self.data_loader)
        cc  = 0 

        for epoch in range(self.start_epoch, config.max_epochs):
            for step, (example, real_im, landmarks, right_audio, wrong_audio) in enumerate(self.data_loader):
                t1 = time.time()

                if config.cuda:
                    real_im    = Variable(real_im).cuda()
                    # landmarks = Variable(landmarks).cuda()
                else:
                    real_im    = Variable(real_im)
                    # landmarks = Variable(landmarks)
                f_im,feature = self.encoder(real_im)


                loss_gen = self.l1_loss_fn(f_im,real_im)

                loss_gen.backward()
                self.opt_g.step()
                self._reset_gradients()

                t2 = time.time()

                if (step+1) % 1 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch
                    eta = int((t2-t1)*steps_remain)

                    print("[{}/{}][{}/{}]   Loss_G: {:.4f} ,  ETA: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch, loss_gen.data[0],  eta))
                    log_value('encoder_loss',loss_gen.data[0] , step + num_steps_per_epoch * epoch)
                if (step ) % (num_steps_per_epoch/3) == 0 :
                    fake_store = f_im.data.permute(0,2,1,3,4).contiguous().view(config.batch_size*16,3,64,64)
                    torchvision.utils.save_image(fake_store, 
                        "{}fake_{}.png".format(config.sample_dir,cc), nrow=16,normalize=True)
                    real_store = real_im.data.permute(0,2,1,3,4).contiguous().view(config.batch_size*16,3,64,64)
                    torchvision.utils.save_image(real_store,
                        "{}real_{}.png".format(config.sample_dir,cc), nrow=16,normalize=True)

                #     fe_store = feature.data.permute(0,2,1,3,4).contiguous().view(config.batch_size*16,3,64,64)
                #     torchvision.utils.save_image(fe_store,
                #         "{}feature_{}.png".format(config.sample_dir,cc), nrow=16,normalize=True)
                    cc += 1
            if epoch % 1 == 0:
                torch.save(self.encoder.state_dict(),
                           "{}/encoder_{}.pth"
                           .format(config.model_dir,epoch))

    def load(self):
        # paths = glob.glob(os.path.join(directory, "*.pth"))
        # gen_path  = [path for path in paths if "encoder" in path][0]
        encoder_path = '/mnt/disk0/dat/lchen63/grid/model/model_embedding/encoder_0.pth'

        self.encoder.load_state_dict(torch.load(encoder_path))

        self.start_epoch = int(encoder_path.split(".")[0].split("_")[-1])
        print("Load pretrained {}".format(encoder_path))

    def _reset_gradients(self):
        self.encoder.zero_grad()
