import cv2
import numpy as np
from dataset import VaganFlowDataset
from flownet.FlowNetS import flownets
from torch.autograd import Variable
import argparse
import torch
import torchvision
import os
import itertools
from torch.utils.data import DataLoader


def visualize_of(flow):
    assert isinstance(flow, np.ndarray)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    h, w, _ = flow.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)
    return bgr

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda",
                        default=True)
    parser.add_argument("--dataset_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/grid/data/pickle/")
    parser.add_argument("--model_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/grid/data/flownets_pytorch.pth")
    parser.add_argument("--sample_dir",
                        type=str,
                        default="/mnt/disk1/dat/lchen63/grid/test_result/vis_flow_no_train/")
    parser.add_argument("--batch_size",
                        type=int,
                        default=8)
    parser.add_argument("--num_thread",
                        type=int,
                        default=40)
    return parser.parse_args()


config = parse_args()
net = flownets(config.model_dir)
net.eval()
net = net.cuda()
testset = VaganFlowDataset(config.dataset_dir, train=False)
test_dataloader = DataLoader(testset,
                                  batch_size=config.batch_size,
                                  num_workers=config.num_thread,
                                  shuffle=False, drop_last=True)

img_idx = 0

for real_im, real_flow in itertools.islice(test_dataloader, 0, 10):
    if config.cuda:
        real_im = Variable(real_im).cuda(async=True)
        real_flow = Variable(real_flow).cuda(async=True)
    else:
        real_im = Variable(real_im)
        real_flow = Variable(real_flow)

    depth = real_im.size()[2]
    for d in range(depth - 1):
        prev_frm = real_im[:, :, d, :, :]
        frm = real_im[:, :, d + 1, :, :]
        proxy_flows = real_flow[:, :, d, :, :].data.cpu()
        gen_flows = net(torch.cat([prev_frm, frm], 1)).data.cpu()

        color_flow_size = list(proxy_flows.size())
        color_gen_flow_size = list(gen_flows.size())
        color_flow_size[1] = 3
        color_gen_flow_size[1] = 3
        color_proxy_flows = torch.zeros(color_flow_size)
        color_gen_flows = torch.zeros(color_gen_flow_size)

        for b in range(config.batch_size):
            proxy_flow = proxy_flows[b]
            gen_flow = gen_flows[b]
            color_proxy_flow = visualize_of(proxy_flow.permute(1, 2, 0).numpy())
            color_gen_flow = visualize_of(gen_flow.permute(1, 2, 0).numpy())
            color_proxy_flow = torch.from_numpy(color_proxy_flow).permute(2, 0, 1)
            color_gen_flow = torch.from_numpy(color_gen_flow).permute(2, 0, 1)
            color_proxy_flows[b] = color_proxy_flow
            color_gen_flows[b] = color_gen_flow

        print('{}: image saved'.format(img_idx))
        torchvision.utils.save_image(prev_frm.data,
                                     os.path.join(config.sample_dir, 'prev_{}.png'.format(img_idx)),
                                     nrow=3)
        torchvision.utils.save_image(frm.data,
                                     os.path.join(config.sample_dir, 'next_{}.png'.format(img_idx)),
                                     nrow=3)
        torchvision.utils.save_image(color_proxy_flows,
                                     os.path.join(config.sample_dir, 'proxy_{}.png'.format(img_idx)),
                                     nrow=3)
        torchvision.utils.save_image(color_gen_flows,
                                     os.path.join(config.sample_dir, 'gen_{}.png'.format(img_idx)),
                                     nrow=3)

        img_idx += 1
