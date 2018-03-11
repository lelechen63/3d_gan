import os
import cv2
from multiprocessing import Pool
import cPickle as pickle
import numpy as np

root = '/mnt/disk1/dat/lchen63/grid/data/regions/'


def worker(vname):
    frames_folder = root + vname
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
    print(vname)
    return vname, mean_flows


if __name__ == '__main__':
    vname_lms = pickle.load(open('/mnt/disk0/dat/zhiheng/lip_movements/grid_trend_lms.pkl'))
    
    pool = Pool(40)
    result = pool.map(worker, vname_lms.keys())
    pickle.dump(result, open('of_result.pkl', 'w+'), True)
