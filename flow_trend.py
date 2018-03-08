import os
import cv2
from multiprocessing import Pool
import cPickle as pickle
import numpy as np


def worker(line):
    video_name = line  # used as key in dict
    line = line.replace('video', 'regions')
    line = line.replace('lipread_vgg', 'lrw')
    frames_folder = line.split('.')[0]
    frame_paths = sorted([f for f in os.listdir(frames_folder) if f.endswith('#lip.jpg')])

    prev = None
    mean_flows = []
    for frame_path in frame_paths:
        cur = cv2.imread(os.path.join(frames_folder, frame_path))
        cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
        cur = cv2.resize(cur, (64, 64))
        if not prev is None:
            flow = cv2.calcOpticalFlowFarneback(prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            displacements = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
            value = np.mean(displacements)
            mean_flows.append(value)                
        prev = cur
    print(video_name)
    return video_name, mean_flows


with open('/mnt/disk1/dat/lchen63/lrw/data/prefix2.txt') as f:
    lines = f.readlines()
    pool = Pool(40)
    result = pool.map(worker, lines)
    pickle.dump(result, open('of_result.pkl', 'w+'), True)
