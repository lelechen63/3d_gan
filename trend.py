import numpy as np
root = '/mnt/disk1/dat/lchen63/lrw/data/'
import pickle
def lms_trend():
	trend = {}
	txt = open(root  + 'prefix.txt','r')
	lmss = []
	count = 0 
	for line in txt:
		line = line[:-1].replace('lipread_vgg', 'lrw')
		temp = line.split('/')
		print temp
		print temp[-1]
		videoname = line
		# lmsname = root +'lms/' + temp[-3] + '/' + temp[-2] + 
		lms_folder_name = line.replace('video','lms')[:-4] 
		previous = None
		tt = []
		for i in range(1,30):
			frame = lms_folder_name + '/' + temp[-1][:-4] + '_%3d.npy'%i
			# previous_f = lms_folder_name + '/' + temp[-1][:-4] + '_%3d.npy'%(i-1)
			if previous == None:
				previous = np.average(np.load(frame))
			else:
				cur = np.average(np.load(frame))
				tt.append(cur - previous)
				previous = cur
		print len(tt)
		print tt
		trend[videoname] = tt
		print count
		count += 1
		if count == 100:
			break
	with open('/mnt/disk1/dat/lchen63/lrw/data/pickle/trend_lms.pkl', 'wb') as handle:
	    pickle.dump(trend, handle, protocol=pickle.HIGHEST_PROTOCOL)

lms_trend()


# 	# for frame_name in sorted(os.listdir( lms_folder_name)):
# 	# 	frame = lms_folder_name + '/' + frame_name


# 	lmss.append( temp[-1][:-3] +'npy')
# previous =None
# previous_v = None
# for name in lmss:
# 	try:
# 		t = np.load(root + 'lms/' + name[:-8] + '/' + name)
# 		node = np.average(np.average(t,axis = 0))
# 		if name[:-8] == previous_v:
# 		    temp.append(node - previous)
# 		else:
# 			if previous_v != None:
# 				trend[previous_v] = temp
# 				# print trend
# 				# print len(temp)
# 				temp = []
# 				previous = node
# 				previous_v = name[:-8] 
# 			else:
# 				temp = []
# 				previous = node
# 				previous_v = name[:-8]
# 	except:
# 		print name

# with open('/mnt/disk1/dat/lchen63/grid/data/trend_lms.pkl', 'wb') as handle:
#     pickle.dump(trend, handle, protocol=pickle.HIGHEST_PROTOCOL)


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






