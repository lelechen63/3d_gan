import numpy as np
root = '/mnt/disk1/dat/lchen63/lrw/data/'
import pickle
import random

def lms_trend():
	trend = {}
	txt = open(root  + 'prefix.txt','r')
	lmss = []
	count = 0

	txt_t = [] 
	for line in txt:
		txt_t.append(line)
	random.shuffle(txt_t)
	new = open(root + 'prefix2.txt','w')
	for i in range(int(len(txt_t)/10)):
		new.write(txt_t[i])

	for line in open(root + 'prefix2.txt','r'):
		line = line[:-1].replace('lipread_vgg', 'lrw')
		temp = line.split('/')
		videoname = line
		lms_folder_name = line.replace('video','lms')[:-4] 
		previous = None
		tt = []
		for i in range(1,30):
			frame = lms_folder_name + '/' + temp[-1][:-4] + '_%03d.npy'%i
			if previous == None:
				previous = np.average(np.load(frame))
			else:
				cur = np.average(np.load(frame))
				tt.append(cur - previous)
				previous = cur
		trend[videoname] = tt
		print count
		count += 1
		if count == 10:
			break
	with open('/mnt/disk1/dat/lchen63/lrw/data/pickle/trend_lms.pkl', 'wb') as handle:
	    pickle.dump(trend, handle, protocol=pickle.HIGHEST_PROTOCOL)

lms_trend()



# ###########################################################################################
# import os
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

# audio = '/mnt/disk1/dat/lchen63/grid/data/trend_lms.pkl'
# op = '/home/zhiheng/lipmotion/gen_flow_distr/of_result.pkl'
# _file = open(audio, "rb")
# audio = pickle.load(_file)

# _file = open(os.path.join(op), "rb")
# op = pickle.load(_file)
# for i in range(len(op)):
#     vname = op[i][0]
#     flow = np.asarray(op[i][1])
#     aud = np.asarray(audio[vname])
#     flow = (flow - flow.min()) / (flow.max() - flow.min())
#     aud= (aud - aud.min()) / (aud.max() - aud.min())
#     x = np.arange(flow.shape[0])
#     print flow.shape
#     print aud.shape
#     print x   
#     plt.plot(x,flow,'r--',x,aud,'g-')
#     plt.savefig('/mnt/disk1/dat/lchen63/grid/data/trend/'+ vname + '.jpg')
#     plt.gcf().clear()






