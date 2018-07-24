import torch
import torch.nn as nn

class FlowsGen(nn.Module):
    def __init__(self, flownet):
        super(FlowsGen, self).__init__()
        self.flownet = flownet
    
    def forward(self, imgs):
        b, c, d, h, w = imgs.size()
        buf = []
        for idx_d in range(d-1):
            prev_im = imgs[:, :, idx_d, :, :]
            cur_im = imgs[:, :, idx_d+1, :, :]
            flow = self.flownet(torch.cat([prev_im, cur_im], 1))[0]
            flow = torch.unsqueeze(flow, 2)
            buf.append(flow)
        buf = torch.cat(buf, 2)
        return buf
        