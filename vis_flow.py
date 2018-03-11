import os
import cv2
import numpy as np


def visualize_of(flow):
    assert isinstance(flow, np.ndarray)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    h, w, _ = flow.shape
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return bgr


root = '/mnt/disk1/dat/lchen63/grid/data/regions/bgbb3a'
vname = 'bgbb3a'

save_root = '/home/zhiheng/lipmotion/vis_flow_only_lips/bgbb3a'

# ABOUT_00001_001.jpg

# for root, subdirs, files in os.walk(root):
#     for file in files:
#         # if file.endswith('.jpg') and not file.endswith('#lip.jpg'):
#         if file.endswith('#lip.jpg'):
#             cur_id = int(file[-11: -8])
#             cur_filepath = os.path.join(root, file)
#             next_filepath = os.path.join(root, file[:-11] + '%03d.jpg' % (cur_id+1))
#             if not os.path.exists(next_filepath):
#                 continue
#             cur_img = cv2.imread(cur_filepath)
#             next_img = cv2.imread(next_filepath)
#             cur_img = cv2.resize(cur_img, (64, 64))
#             next_img = cv2.resize(next_img, (64, 64))
#             cur_img = cv2.cvtColor(cur_img, cv2.COLOR_BGR2GRAY)
#             next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
#
#             flow = cv2.calcOpticalFlowFarneback(cur_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#             bgr = visualize_of(flow)
#
#             cv2.imwrite(os.path.join(save_root, file), bgr)


for i in range(1, 76):
    fname = vname + '_%03d#lip.jpg' % i
    fpath = os.path.join(root, fname)
    assert os.path.exists(fpath)

    prev = None
    cur = cv2.imread(fpath)
    cur = cv2.resize(cur, (64, 64))
    cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    if prev is not None:
        flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        bgr = visualize_of(flow)
        cv2.imwrite(os.path.join(save_root, fname), bgr)

    prev = cur
