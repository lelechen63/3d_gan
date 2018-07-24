import pickle
import random
import os
import scipy.io.wavfile
import librosa
import math
import numpy as np
import argparse
from imutils import face_utils
import imutils
import dlib
import cv2
import multiprocessing
from sklearn.decomposition import PCA

def parse_arguments():
    """Parse arguments from command line"""
    description = "Train a model."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    parser.add_argument('--root_path', '-p',
        default="/mnt/disk0/dat/lchen63/grid/data/",
        help = 'data path'
        )
    parser.add_argument('--shape_predictor', '-sp',
        default="/home/lchen63/project/text-to-image.pytorch/data/shape_predictor_68_face_landmarks.dat",
        help = 'data path'
        )

    return parser.parse_args()
args = parse_arguments()
path = args.root_path


def generate_txt():
	filenames = os.listdir(path + 'align/')
	txt = open(path + 'prefix.txt','w')
	for filename in filenames:
		print filename
		file_prefix = filename[:-6]
		print file_prefix
		txt.write(file_prefix + '\n')
	print len(filenames)


def extract_images():
	txt = open(path + 'prefix.txt','r')
	count = 0
	for line in txt:
		# count += 1
		# if count == 2:
		# 	break
		if not os.path.exists(path + 'image/' + line[:-1]):
			os.mkdir(path + 'image/' + line[:-1])
		if not os.path.exists(path + 'landmark/' + line[:-1]):
			os.mkdir(path + 'landmark/' + line[:-1])
		if not os.path.exists(path + 'chunk/' + line[:-1]):
			os.mkdir(path + 'chunk/' + line[:-1])
		if not os.path.exists(path + 'lips/' + line[:-1]):
			os.mkdir(path + 'lips/' + line[:-1])

		new_video = path + 'video/' + line[:-1] + '.mpg'

		command = ' ffmpeg -i ' + new_video + ' ' + path + 'image/' + line[:-1] + '/' + line[:-1] + '_%03d.jpg'
		print command
		try:
			os.system(command)
		except:
			print  new_video
# extract_images()

def crop_lips(image_path):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args.shape_predictor)

	try:
		# load the input image, resize it, and convert it to grayscale
		image = cv2.imread(image_path)
		# image = imutils.resize(image, width=500)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		 
		# detect faces in the grayscale image
		rects = detector(gray, 1)
		for (i, rect) in enumerate(rects):

			
			shape = predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)
		 
			for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
				if name != 'mouth':
					continue

				(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

				center_x = x + int(0.5*w)

				center_y = y + int(0.5*h)

				if w > h:
					r  = int(0.65 * w)
				else:
					r = int(0.65 * h)
				new_x = center_x - r
				new_y = center_y - r
				roi = image[new_y:new_y + 2 * r , new_x:new_x + 2 * r]

				return shape,roi
	except:
		print  image_path

global CC
CC = 0 

def _extract_audio(lists):
	global CC
	CC = 0
	for line in lists:
		CC += 1
		print '++++++++++++++++' + str(CC) + '/' + str(len(lists))
		temp = line.split('/')
		if not os.path.exists(path + 'audio/' + temp[-2]):
			os.mkdir(path + 'audio/' + temp[-2])
		command = 'ffmpeg -i ' + line + '  -ac 1  ' +  path + 'audio/'  + temp[-1][:-4] + '.wav'
		print command
		try:
			os.system(command)
		except:
			print  line
def extract_audio():
	txt = open(path + 'prefix.txt','r')
	count = 0
	if not os.path.exists(path + 'audio/'):
		os.mkdir(path + 'audio/' )
	total = []
	for line in txt:
		total.append(path + 'video/' + line[:-1] + '.mpg')
	batch = 1
	datas = []
	batch_size = len(total)/ batch
	temp = []
	for i,d in enumerate(total):
		temp.append(d)
		if (i+1) % batch_size == 0:
			datas.append(temp)
			temp= []
	print len(datas)

	for i in range(batch):
		process = multiprocessing.Process(target = _extract_audio,args = (datas[i],))
        process.start()

def wav2mfcc(wav = None,sr = 44100):
	y = wav
	
	S = librosa.feature.mfcc(y, sr=sr,n_mfcc=12,fmax= 16000)
	return S


def wav2lms(wav = None,sr = 44100):
	y = wav
	
	S = librosa.feature.melspectrogram(y, sr=sr,n_fft = 1024, n_mels=128,fmax= 16000)
	log_S = librosa.logamplitude(S)
	return log_S

def generate_lms():
	print 'ggg'
	lms_txt = open(path + 'lms.txt','w')
	if not os.path.exists(path + 'lms/'):
		os.mkdir(path + 'lms/' )
	if not os.path.exists(path + 'chunk/'):
		os.mkdir(path + 'chunk/' )
	txt = open(path + 'prefix.txt','r')
	for line in txt:
		if not os.path.exists(path + 'lms/' + line[:-1]):
			os.mkdir(path + 'lms/' + line[:-1])
		if not os.path.exists(path + 'chunk/' + line[:-1]):
			os.mkdir(path + 'chunk/' + line[:-1])
		audio_path = path + 'audio/'  + line[:-1] + '.wav'
		try:
			sr, wav = scipy.io.wavfile.read(audio_path)
			total_lenth = wav.shape[0]
			print total_lenth
			chunk_lenth = total_lenth/75.0
			print chunk_lenth
			for i in range(0,75):
				y = wav[int(i*chunk_lenth):int(i*chunk_lenth + chunk_lenth)]
				lms = wav2lms(y,sr)
				lms_chunk_name = path + 'lms/' + line[:-1] + '/' +  line[:-1] + '_%03d.npy' %(i)
				chunk_name = path + 'chunk/' + line[:-1] + '/' +  line[:-1] + '_%03d.npy' %(i)
				if  lms.shape != (128,4):
					print audio_path
					print '++'
					print lms.shape
					break
				np.save(lms_chunk_name,lms)
				np.save(chunk_name,y)
				lms_txt.write(line[:-1] +'_%03d' %(i) )
				
		except:
			print audio_path
def generate_mfcc():
	print 'ggg'
	mfcc_txt = open(path + 'mfcc.txt','w')
	if not os.path.exists(path + 'mfcc/'):
		os.mkdir(path + 'mfcc/' )
	txt = open(path + 'prefix.txt','r')
	for line in txt:
		if not os.path.exists(path + 'mfcc/' + line[:-1]):
			os.mkdir(path + 'mfcc/' + line[:-1])
		audio_path = path + 'audio/'  + line[:-1] + '.wav'
		try:
			sr, wav = scipy.io.wavfile.read(audio_path)
			total_lenth = wav.shape[0]
			print total_lenth
			chunk_lenth = total_lenth/75.0
			print chunk_lenth
			for i in range(0,75):
				y = wav[int(i*chunk_lenth):int(i*chunk_lenth + chunk_lenth)]
				mfcc = wav2mfcc(y,sr)
				mfcc_chunk_name = path + 'mfcc/' + line[:-1] + '/' +  line[:-1] + '_%03d.npy' %(i)
				if  mfcc.shape != (12,4):
					print audio_path
					print '++'
					print mfcc.shape
					break
				np.save(mfcc_chunk_name,mfcc)
				mfcc_txt.write(line[:-1] +'_%03d' %(i) )
				
		except:
			print audio_path



def generating_landmark_lips(lists):
	# image_txt = open(path + 'image.txt','r')
	image_txt = lists
	land_shape = {}
	lip_shape = {}
	for line in image_txt:
		img_path = line
		temp = img_path.split('/')
		if os.path.isfile(path + 'lips/' +  temp[-2] + '/' +temp[-1][:-4] + '.jpg') and  os.path.isfile( path + 'landmark'  + '/' +temp[-2] + '/' +temp[-1][:-4] + '.npy'):
			print '---'
			print temp[-1]
			continue

		if not os.path.exists(path + 'lips/' +temp[-2]):
			os.mkdir(path + 'lips/' + temp[-2])
		if not os.path.exists(path + 'landmark/' +temp[-2]):
			os.mkdir(path + 'landmark/' + temp[-2])
		landmark_path = path + 'landmark'  + '/' +temp[-2] + '/' +temp[-1][:-4] + '.npy'
		lip_path = path + 'lips/' +  temp[-2] + '/' +temp[-1][:-4] + '.jpg'
		try:
			landmark, lip = crop_lips(img_path)
			print lip.shape
			print landmark.shape
			cv2.imwrite(lip_path,lip)
			np.save(landmark_path,landmark)

		except:
			print line
def multi_pool():
    data = []
    for root, dirs, files in os.walk(path + 'image/'):
		for file in files:
			if file[-3:] == 'jpg':

				name =  os.path.join(root, file)
				print name
				data.append(name)
    num_thread = 40
    datas = []
    batch_size = int(len(data)/num_thread)
    temp = []
    for  i,d in enumerate (data):
        temp.append(d)
        if (i + 1) % batch_size == 0:
            datas.append(temp)
            temp = []
    print len(datas)
    for i in range(num_thread):
        process = multiprocessing.Process(target = generating_landmark_lips,args = (datas[i],))
        process.start()

def get_x():
    root = '/mnt/disk1/dat/lchen63/grid/data/landmark'
    landmark = []
    names = []
    gg =0
    for path, subdirs, files in os.walk(root):

        for name in files:
            lmname =  os.path.join(path, name)
            lm = np.load(lmname)[48:]
            original = np.sum(lm,axis=0) / 20.0
            lm = lm - original
            lm = np.reshape(lm,40)
            landmark.append(lm)
            names.append(name)
    landmark = np.asarray(landmark)
    print  '======================'
    print landmark.shape
    return landmark,names

def get_pca():
    pca_root = '/mnt/disk1/dat/lchen63/grid/data/pca_landmark/'
    pca = PCA(n_components = 16)
    points, paths = get_x()
    pca.fit(points)
    new_points = pca.transform(points)
    for i,name in enumerate(paths):
        if not os.path.exists(pca_root + name.split('_')[0]):
            os.mkdir(pca_root + name.split('_')[0])
        lm_p = pca_root + name.split('_')[0] + '/' + name
        np.save(lm_p, new_points[i,:])



def get_data():
	print path + 'image.txt'
	data_txt = open( path + 'image.txt')
	data_information = []

	count = 0

	for line in data_txt:
		image_path = line[:-1]
		temp = image_path.split('/')
		data_information.append( temp[-2] + '/' + temp[-1][:-4] )
	print len(data_information)
	return data_information






def generate_video_pickle():
	datalists = []
	for path, subdirs, files in os.walk('/mnt/disk0/dat/lchen63/grid/data/lips'):
	    for name in files:
	        datalists.append( os.path.join(path, name))
	data = []
	count = 0
	s33_testing_data = []
	filenames = os.listdir('/mnt/disk0/dat/lchen63/grid/tmp/s33_align/')
	s33 = set()
	for filename in filenames:
		s33.add(filename[:-6])
	for i in xrange(0,len(datalists),8):
		fff = {}
		temp = datalists[i].split('/')
		start_fram = int(temp[-1].split('.')[0].split('_')[-1])
		img_path = args.root_path + 'lips/' + temp[-2] 
		lms_path = args.root_path + 'lms/' +  temp[-2]
		imgs = []
		lmss = []
		for j in range(0,16):
			img_f = img_path + '/' + temp[-2] + '_%03d.jpg'%(j + start_fram) 
			lms_f = lms_path + '/' + temp[-2] + '_%03d.npy'%(j + start_fram)
			if  os.path.isfile(img_f) and os.path.isfile(lms_f) :
				imgs.append(img_f)
				lmss.append(lms_f)
			else:

				print img_f
				print lms_f
				count += 1
				break
			if np.load(lms_f).shape != (128,4):
				print 'gg'
				print np.load(lms_f).shape
				break
			fff["image_path"]= imgs
			fff["lms_path"]= lmss
			if j == 15:
				if temp[-2] in s33:
					s33_testing_data.append(fff)
				else:
					data.append(fff)

	print count
	print len(data)
	print len(s33_testing_data)
	print 'training:\t'+ str(int(len(data)*0.9))
	print 'testing:\t' + str(len(data) - int(len(data)*0.9))
	train_data = data[:int(0.9*len(data))]
	test_data = data[int(0.9*len(data)):]
	random.shuffle(train_data)
	random.shuffle(test_data)
	random.shuffle(s33_testing_data)
	with open('/mnt/disk0/dat/lchen63/grid/data/pickle/train.pkl', 'wb') as handle:
		pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open('/mnt/disk0/dat/lchen63/grid/data/pickle/test.pkl', 'wb') as handle:
		pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open('/mnt/disk0/dat/lchen63/grid/data/pickle/new_test.pkl', 'wb') as handle:
		pickle.dump(s33_testing_data, handle, protocol=pickle.HIGHEST_PROTOCOL)




# generate_txt()
# extract_images()
# delete_silence()
# extract_audio()
# generate_lms()
# generate_mfcc()
# multi_pool()
generate_video_pickle()
