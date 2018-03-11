from __future__ import print_function
import numpy as np
root = '/mnt/disk1/dat/lchen63/lrw/data/'
import cPickle as pickle
from multiprocessing import Pool
import os


def worker(line):
    line = line[:-1].replace('lipread_vgg', 'lrw')
    temp = line.split('/')
    videoname = line
    lms_folder_name = line.replace('video', 'lms')[:-4]

    previous = None
    tt = []
    for i in range(1, 30):
        frame = lms_folder_name + '/' + temp[-1][:-4] + '_%03d.npy' % i
        if not os.path.exists(frame):
            return videoname, None
        if previous == None:
            previous = np.average(np.load(frame))
        else:
            cur = np.average(np.load(frame))
            if np.any(np.isinf(cur)):
                has_inf = True
                return videoname, None
            tt.append(cur - previous)
            previous = cur

	print(videoname)
	return videoname, tt
    			
    	
if __name__ == '__main__':
    with open(root + 'prefix.txt', 'r') as f:
        lines = f.readlines()
        result = dict
        # for line in lines:
    # 	    videoname, tt = worker(line)
        #     result[videoname, tt]
        pool = Pool(40)
        result = pool.map(worker, lines)
        result = dict([(vname, tt)
                       for (vname, tt) in result if tt is not None])

    # if count == 10:
    # 	break
    with open('/mnt/disk0/dat/zhiheng/lip_movements/trend_lms.pkl', 'wb') as handle:
        pickle.dump(result, handle)
