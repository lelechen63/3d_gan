import pickle
import numpy as np
a = '/mnt/disk1/dat/lchen63/lrw/data/pickle/new_video_16_train.pkl'
_file = open(a, "rb")
train_data = pickle.load(_file)
count = 0
total = len(train_data)
for index in range(total):
	f =0
	for i in  range(0,16):
		lms_path = train_data[index][1 + i*3]
		melFrames = np.transpose(np.load(lms_path))
		# print melFrames[0][0]
        if np.isinf(melFrames[0][0]):
            f = 1
            
    if f == 1:
    	count += 1 
    print '{}/{}'.format(count,total)
