from FlowNetS import flownets
import os
import torch
import torch.utils
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import VaganFlowDataset
from tensorboard_logger import configure, log_value
from flownet.multiscaleloss import multiscaleloss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer():
    def __init__(self, config):
        self.net = flownets(config.flownet_pth)

        self.trainset = VaganFlowDataset(config.dataset_dir, train=True)
        self.testset = VaganFlowDataset(config.dataset_dir, train=False)

        self.train_dataloader = DataLoader(self.trainset,
                                           batch_size=config.batch_size,
                                           num_workers=config.num_thread,
                                           shuffle=True, drop_last=True)
        self.test_dataloader = DataLoader(self.testset,
                                          batch_size=config.batch_size,
                                          num_workers=config.num_thread,
                                          shuffle=False, drop_last=True)

        self.optimizer = torch.optim.Adam(self.net.parameters(), config.lr,
                                betas=(0.9, 0.999), weight_decay=config.weight_decay)
        self.criterion = multiscaleloss(sparse=False, loss='L1')
        self.high_res_EPE = multiscaleloss(scales=1, downscale=4, weights=(1), loss='L1',
                                      sparse=False)
        if config.cuda:
            device_ids = [int(i) for i in config.device_ids.split(',')]
            self.net = nn.DataParallel(self.net.cuda(), device_ids=device_ids)
            self.criterion = self.criterion.cuda()
            self.high_res_EPE = self.high_res_EPE.cuda()

        self.config = config
        self.start_epoch = 0

    def fit(self):
        config = self.config
        configure("{}".format(config.log_dir), flush_secs=5)
        depth = 15
        num_iter = config.max_epochs * len(self.train_dataloader) * depth

        for epoch in range(self.start_epoch, config.max_epochs):
            train_losses = AverageMeter()
            train_flow2_EPEs = AverageMeter()
            self.net.train()
            for step, (real_im, real_flow) in enumerate(self.train_dataloader):
                if config.cuda:
                    real_im = Variable(real_im).cuda(async=True)
                    real_flow = Variable(real_flow).cuda(async=True)
                else:
                    real_im = Variable(real_im)
                    real_flow = Variable(real_flow)

                depth = real_im.size()[2]
                for d in range(depth-1):
                    prev_frm = real_im[:, :, d, :, :]
                    frm = real_im[:, :, d+1, :, :]
                    proxy_flow = real_flow[:, :, d, :, :]
                    gen_flows = self.net(torch.cat([prev_frm, frm], 1))

                    loss = self.criterion(gen_flows, proxy_flow)
                    flow2_EPE = 20 * self.high_res_EPE(gen_flows[0], proxy_flow)

                    train_losses.update(loss.data[0], real_flow.size(0))
                    train_flow2_EPEs.update(flow2_EPE.data[0], real_flow.size(0))

                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    iter_idx = epoch*(len(self.train_dataloader) * depth) + step * depth + d
                    print('epoch: {}/{} iter: {}/{}, loss: {}, EPE: {}'.
                          format(epoch, config.max_epochs, iter_idx, num_iter, loss.data[0], flow2_EPE.data[0]))
                    log_value('loss', loss.data[0], iter_idx)
                    log_value('EPE', flow2_EPE.data[0], iter_idx)

            log_value('loss_epoch', train_losses.avg, epoch)
            log_value('EPE_epoch', train_flow2_EPEs.avg, epoch)

            self.adjust_learning_rate(self.optimizer, epoch*depth)
            torch.save(self.net.state_dict(),
                      os.path.join(config.model_dir, 'flownet_{}.pth'.format(epoch)))

            # tesing
            test_flow2_EPEs = AverageMeter()
            self.net.eval()
            for step, (real_im, real_flow) in enumerate(self.test_dataloader):
                if config.cuda:
                    real_im = Variable(real_im).cuda(async=True)
                    real_flow = Variable(real_flow).cuda(async=True)
                else:
                    real_im = Variable(real_im)
                    real_flow = Variable(real_flow)

                depth = real_im.size()[2]
                for d in range(depth-1):
                    prev_frm = real_im[:, :, d, :, :]
                    frm = real_im[:, :, d+1, :, :]
                    proxy_flow = real_flow[:, :, d, :, :]
                    gen_flow = self.net(torch.cat([prev_frm, frm], 1))

                    flow2_EPE = 20 * self.high_res_EPE(gen_flow, proxy_flow)
                    test_flow2_EPEs.update(flow2_EPE.data[0])

                    print('epoch: {}/{} avg_EPE: {}'.
                          format(epoch, config.max_epochs, test_flow2_EPEs.avg))
            log_value('test_EPE_epoch', test_flow2_EPEs.avg, epoch)

    def adjust_learning_rate(self, optimizer, epoch):
        """Sets the learning rate to the initial LR decayed by 2 after 300K iterations, 400K and 500K"""
        if epoch == 10 or epoch == 15 or epoch == 20:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 2
