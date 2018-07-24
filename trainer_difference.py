import os
import glob
import time
import torch
import torch.utils
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import VaganDataset,LRWdataset
from model_difference import Generator, Discriminator
from tensorboard_logger import configure, log_value
from collections import OrderedDict

class Trainer():
    def __init__(self, config):
        self.generator = Generator()
        self.discriminator = Discriminator()

        print(self.generator)
        print(self.discriminator)
        self.bce_loss_fn = nn.BCELoss()
        self.mse_loss_fn =  nn.MSELoss()

        self.opt_g = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator.parameters()),
            lr=config.lr, betas=(config.beta1, config.beta2))
        self.opt_d = torch.optim.Adam(self.discriminator.parameters(),
            lr=config.lr, betas=(config.beta1, config.beta2))

        if config.dataset == 'grid':
            self.dataset = VaganDataset(config.dataset_dir, train=config.is_train)
        elif config.dataset == 'lrw':
            self.dataset = LRWdataset(config.dataset_dir, train=config.is_train)

        self.data_loader = DataLoader(self.dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_thread,
                                      shuffle=True, drop_last=True)
        data_iter = iter(self.data_loader)
        data_iter.next()
        self.ones = Variable(torch.ones(config.batch_size), requires_grad=False)
        self.zeros = Variable(torch.zeros(config.batch_size), requires_grad=False)

        device_ids = [int(i) for i in config.device_ids.split(',')]
        if config.cuda:
            self.generator     = nn.DataParallel(self.generator.cuda(), device_ids=device_ids)
            self.discriminator = nn.DataParallel(self.discriminator.cuda(), device_ids=device_ids)
            self.bce_loss_fn   = self.bce_loss_fn.cuda()
            self.mse_loss_fn = self.mse_loss_fn.cuda()
            self.ones          = self.ones.cuda()
            self.zeros         = self.zeros.cuda()

        self.config = config
        self.start_epoch = 0
        self.load(config.model_dir)

    def fit(self):
        config = self.config
        configure("{}/".format(config.log_dir), flush_secs=5)

        num_steps_per_epoch = len(self.data_loader)
        cc  = 0


        for epoch in range(self.start_epoch, config.max_epochs):
            for step, (example, real_im, landmarks, right_audio, wrong_audio) in enumerate(self.data_loader):

                t1 = time.time()

                if config.cuda:
                    example = Variable(example).cuda()
                    landmarks = Variable(landmarks).cuda()
                    real_im    = Variable(real_im).cuda()
                    right_audio = Variable(right_audio).cuda()
                    wrong_audio = Variable(wrong_audio).cuda()
                else:
                    example = Variable(example)
                    landmarks = Variable(landmarks)
                    real_im    = Variable(real_im)
                    right_audio = Variable(right_audio)
                    wrong_audio = Variable(wrong_audio)


                fake_im = self.generator(example, right_audio)

                # Train the discriminator
                D_real  = self.discriminator(real_im, right_audio)
                D_wrong = self.discriminator(real_im, wrong_audio)
                D_fake  = self.discriminator(fake_im.detach(), right_audio)


                loss_real  = self.bce_loss_fn(D_real, self.ones)
                loss_wrong = self.bce_loss_fn(D_wrong, self.zeros)
                loss_fake  = self.bce_loss_fn(D_fake, self.zeros)

                loss_disc = loss_real + 0.5*(loss_fake + loss_wrong)
                loss_disc.backward()
                self.opt_d.step()
                self._reset_gradients()

                # Train the generator
                fake_im = self.generator(example, right_audio)
                D_fake  = self.discriminator(fake_im, right_audio)
                loss_gen   = self.bce_loss_fn(D_fake, self.ones)

                loss_gen.backward()
                self.opt_g.step()
                self._reset_gradients()

                t2 = time.time()

                if (step+1) % 1 == 0 or (step+1) == num_steps_per_epoch:
                    steps_remain = num_steps_per_epoch-step+1 + \
                        (config.max_epochs-epoch+1)*num_steps_per_epoch
                    eta = int((t2-t1)*steps_remain)

                    print("[{}/{}][{}/{}] Loss_D: {:.4f}  Loss_G: {:.4f},  ETA: {} second"
                          .format(epoch+1, config.max_epochs,
                                  step+1, num_steps_per_epoch,
                                  loss_disc.data[0], loss_gen.data[0],  eta))
                    log_value('discriminator_loss',loss_disc.data[0] , step + num_steps_per_epoch * epoch)
                    log_value('generator_loss',loss_gen.data[0] , step + num_steps_per_epoch * epoch)
                if (step ) % (num_steps_per_epoch/3) == 0 :
                    fake_store = fake_im.data.permute(0,2,1,3,4).contiguous().view(config.batch_size*16,3,64,64)
                    torchvision.utils.save_image(fake_store,
                        "{}fake_{}.png".format(config.sample_dir,cc), nrow=16,normalize=True)
                    real_store = real_im.data.permute(0,2,1,3,4).contiguous().view(config.batch_size*16,3,64,64)
                    torchvision.utils.save_image(real_store,
                        "{}real_{}.png".format(config.sample_dir,cc), nrow=16,normalize=True)
                    cc += 1
            if epoch % 1 == 0:
                torch.save(self.generator.state_dict(),
                           "{}/generator_{}.pth"
                           .format(config.model_dir,epoch))
                torch.save(self.discriminator.state_dict(),
                           "{}/discriminator_{}.pth"
                           .format(config.model_dir, epoch))

    def load(self, directory):
        # paths = glob.glob(os.path.join(directory, "*.pth"))
        # gen_path  = [path for path in paths if "generator" in path][0]
        # disc_path = [path for path in paths if "discriminator" in path][0]
        gen_path = '/mnt/disk0/dat/lchen63/grid/model/model_difference/generator_5.pth'
        disc_path = '/mnt/disk0/dat/lchen63/grid/model/model_difference/discriminator_5.pth'
        print("Loading pretrained [{}, {}]".format(gen_path, disc_path))

        # gen_state_dict = torch.load(gen_path)
        # new_gen_state_dict = OrderedDict()
        # for k, v in gen_state_dict.items():
        #     name = 'model.' + k 
        #     new_gen_state_dict[name] = v
        # # load params
        # self.generator.load_state_dict(new_gen_state_dict)

        # disc_state_dict = torch.load(disc_path)
        # new_disc_state_dict = OrderedDict()
        # for k, v in disc_state_dict.items():
        #     name = 'model.' + k 
        #     new_disc_state_dict[name] = v
        # # load params
        # self.discriminator.load_state_dict(new_disc_state_dict)
        self.generator.load_state_dict(torch.load(gen_path))
        self.discriminator.load_state_dict(torch.load(disc_path))

        self.start_epoch = int(gen_path.split(".")[0].split("_")[-1])
        print self.start_epoch

        

    def _reset_gradients(self):
        self.generator.zero_grad()
        self.discriminator.zero_grad()
