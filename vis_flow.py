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


root = '/mnt/disk1/dat/lchen63/lrw/data/regions'
save_root = '/home/zhiheng/lipmotion/vis_flow_full_face'

# ABOUT_00001_001.jpg

for root, subdirs, files in os.walk(root):
    for file in files:
        if file.endswith('.jpg') and not file.endswith('#lip.jpg'):
            cur_id = int(file[-7, -4])
            if cur_id > 29:
                continue
            cur_filepath = os.path.join(root, file)
            next_filepath = os.path.join(root, file[:-7] + '%03d.jpg' % cur_id+1)
            cur_img = cv2.imread(cur_filepath, cv2.IMREAD_GRAYSCALE)
            next_img = cv2.imread(next_filepath, cv2.IMREAD_GRAYSCALE)
            flow = cv2.calcOpticalFlowFarneback(cur_img, next_img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            bgr = visualize_of(flow)

            cv2.imwrite(os.path.join(save_root, file), bgr)
            
