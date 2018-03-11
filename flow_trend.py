import os
import cv2
from multiprocessing import Pool
import cPickle as pickle
import numpy as np

root = '/mnt/disk1/dat/lchen63/grid/data/regions/'


def worker(vname):
    frames_folder = root + vname
    prev = None
    mean_flows = []
    for i in range(1, 76):
        fname = vname + '_%03d#lip.jpg' % i
        frame_path = os.path.join(frames_folder, fname)
        if not os.path.exists(frame_path):
            print('frame path not exists: {}'.format(frame_path))
            return vname, None
        cur = cv2.imread(frame_path)
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
    result = dict([(k,v) for k,v in result if v is not None and len(v) == 74])
    pickle.dump(result, open('of_result.pkl', 'w+'), True)
