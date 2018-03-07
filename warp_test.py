from torch.autograd import Variable
import torch
import numpy as np
import cv2
from ops import Warp

def load_flo(path):
    with open(path, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        assert(202021.25 == magic),'Magic number incorrect. Invalid .flo file'
        w = np.fromfile(f, np.int32, count=1)[0]
        h = np.fromfile(f, np.int32, count=1)[0]
        data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    data2D = np.resize(data, (h, w, 2))
    return data2D

flow_path = '/Users/hubert/18298_flow.flo'
img1_path = '/Users/hubert/18298_img2.ppm'

flow = load_flo(flow_path)
flow[:, :, 0], flow[:, :, 1] = flow[:, :, 1].copy(), flow[:, :, 0].copy()

flow = np.expand_dims(flow, 0)
flow = np.transpose(flow, (0, 3, 1, 2))
flow = Variable(torch.from_numpy(flow).type(torch.FloatTensor))

img1 = cv2.imread(img1_path)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img1 = np.transpose(img1, (2, 0, 1))  # HWC ==> CHW
img1 = np.expand_dims(img1, 0)  # CHW ==> BCHW
img1 = torch.from_numpy(img1).type(torch.FloatTensor)
img1 = Variable(img1)

warp = Warp(flow.size())

output = warp(img1, flow)
img2 = output.data.numpy()
img2 = np.transpose(img2[0], (1, 2, 0)).astype(np.uint8)
img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

cv2.imshow('img2', img2)
cv2.waitKey()
