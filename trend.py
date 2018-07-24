import numpy as np
root = '/mnt/disk1/dat/lchen63/grid/data/'
import pickle


train = '/mnt/disk1/dat/lchen63/grid/data/pickle/train.pkl'
test = '/mnt/disk1/dat/lchen63/grid/data/pickle/test.pkl'
test2 = '/mnt/disk1/dat/lchen63/grid/data/pickle/s33_test.pkl'

# _file = open(train, "rb")
# train = pickle.load(_file)

# _file = open(test, "rb")
# test = pickle.load(_file)

# _file = open(test2, "rb")
# test2 = pickle.load(_file)
# print len(train)
# print len(test)
# print len(test2)
trend = {}
txt = open(root  + 'image.txt','r')
lmss = []
for line in txt:
	line = line[:-1]
	temp = line.split('/')
	videoname = temp[-2]
	lmss.append( temp[-1][:-3] +'npy')
previous =None
previous_v = None
for name in lmss:
	try:
		t = np.load(root + 'lms/' + name[:-8] + '/' + name)
		node = np.average(np.average(t,axis = 0))
		if name[:-8] == previous_v:
		    temp.append(node - previous)
		else:
			if previous_v != None:
				trend[previous_v] = temp
				# print trend
				# print len(temp)
				temp = []
				previous = node
				previous_v = name[:-8] 
			else:
				temp = []
				previous = node
				previous_v = name[:-8]
	except:
		print name

with open('/mnt/disk1/dat/lchen63/grid/data/trend_lms.pkl', 'wb') as handle:
    pickle.dump(trend, handle, protocol=pickle.HIGHEST_PROTOCOL)


###########################################################################################
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

audio = '/mnt/disk1/dat/lchen63/grid/data/trend_lms.pkl'
op = '/home/zhiheng/lipmotion/gen_flow_distr/of_result.pkl'
_file = open(audio, "rb")
audio = pickle.load(_file)

_file = open(os.path.join(op), "rb")
op = pickle.load(_file)
for i in range(len(op)):
    vname = op[i][0]
    flow = np.asarray(op[i][1])
    aud = np.asarray(audio[vname])
    flow = (flow - flow.min()) / (flow.max() - flow.min())
    aud= (aud - aud.min()) / (aud.max() - aud.min())
    x = np.arange(flow.shape[0])
    print flow.shape
    print aud.shape
    print x   
    plt.plot(x,flow,'r--',x,aud,'g-')
    plt.savefig('/mnt/disk1/dat/lchen63/grid/data/trend/'+ vname + '.jpg')
    plt.gcf().clear()






