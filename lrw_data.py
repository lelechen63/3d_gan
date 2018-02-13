import pickle
import os
import scipy.io.wavfile
from skimage import data

import librosa
import numpy as np
import argparse
from imutils import face_utils
import dlib
import cv2
import multiprocessing
import matplotlib.pyplot as plt
import scipy.ndimage as sp
from skimage import transform as tf

def parse_arguments():
    """Parse arguments from command line"""
    description = "Train a model."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--root_path', '-p',
                        # default="/mnt/disk1/dat/lchen63/lrw/data",
                        default='/media/lele/DATA/lrw/data',
                        help='data path'
                        )
    
    # parser.add_argument(
    #     '--data_dir',
    #     '-data_dir',
    #     default='/mnt/disk1/dat/lchen63/lrw/data',
    #     # default='/media/lele/DATA/lrw/data2',
    #     help = 'data_dir')

    return parser.parse_args()


args = parse_arguments()
global CC
CC = 0
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(args.root_path,'shape_predictor_68_face_landmarks.dat'))

def generate_txt():
    txt = open(os.path.join(args.root_path , 'prefix.txt'), 'w')
    for root, dirs, files in os.walk(os.path.join(args.root_path , 'video')):
        for file in files:
            if file[-3:] == 'mp4':
                name = os.path.join(root, file)
                txt.write(name + '\n')


def _extract_images(lists):
    for line in lists:
        temp = line.split('/')
        if not os.path.exists(os.path.join(os.path.join(args.root_path , 'image'),temp[-3])):
            os.mkdir(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]))
        if not os.path.exists(os.path.join(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]), temp[-2])):
            os.mkdir(os.path.join(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]), temp[-2]))


        
        if not os.path.exists(
                os.path.join(os.path.join(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]), temp[-2]), temp[-1][:-4])):
            os.mkdir(os.path.join(os.path.join(os.path.join(os.path.join(args.root_path , 'image'),temp[-3]), temp[-2]), temp[-1][:-4]))

        command = ' ffmpeg -i ' + line + ' -r 25 ' + line.replace('video','image').replace('.mp4','/') +  temp[-1][:-4] + '_%5d.jpg'
        print command
        try:
            # pass
            os.system(command)
        except BaseException:
            print line
    # else:
    #     continue


def extract_images():
    txt = open(os.path.join(args.root_path , 'prefix.txt'), 'r')
    if not os.path.exists(os.path.join(args.root_path ,  'image')):
        os.mkdir(os.path.join(args.root_path ,  'image'))
    total = []
    for line in txt:
        total.append(line[:-1])
    batch = 1
    datas = []
    batch_size = len(total) / batch
    temp = []
    for i, d in enumerate(total):
        temp.append(d)
        if (i + 1) % batch_size == 0:
            datas.append(temp)
            temp = []
    print len(datas)

    for i in range(batch):
        process = multiprocessing.Process(
            target=_extract_images, args=(datas[i],))
    process.start()


def _extract_audio(lists):
    for line in lists:
        temp = line.split('/')
        if not os.path.exists(os.path.join(os.path.join(args.root_path , 'audio'),temp[-3])):
            os.mkdir(os.path.join(os.path.join(args.root_path , 'audio'),temp[-3]))
        if not os.path.exists(os.path.join(os.path.join(os.path.join(args.root_path , 'audio'),temp[-3]), temp[-2])):
            os.mkdir(os.path.join(os.path.join(os.path.join(args.root_path , 'audio'),temp[-3]), temp[-2]))

        command = 'ffmpeg -i ' + line + ' -ar 44100  -ac 1  ' + line.replace('video','audio').replace('.mp4','.wav') 
        print command
        try:
            # pass
            os.system(command)
        except BaseException:
            print line


def extract_audio():
    txt = open(os.path.join(args.root_path , 'prefix.txt'), 'r')
    if not os.path.exists(os.path.join(args.root_path , 'audio')):
        os.mkdir(os.path.join(args.root_path , 'audio'))
    total = []
    for line in txt:
        total.append(line[:-1])
    batch = 1
    datas = []
    batch_size = len(total) / batch
    temp = []
    for i, d in enumerate(total):
        temp.append(d)
        if (i + 1) % batch_size == 0:
            datas.append(temp)
            temp = []
    print len(datas)
    _extract_audio(datas[0])

    # for i in range(batch):
    #     process = multiprocessing.Process(target = _extract_audio,args = (datas[i],))
 #        process.start()





def get_base_roi(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(args.root_path,'shape_predictor_68_face_landmarks.dat'))

    # try:
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    # new_im= cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)
    # cv2.imwrite(image_path,new_im)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        print shape
        print shape.shape

        (x, y, w, h) = face_utils.rect_to_bb(rect)
        print x,y,w,h
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)
        print center_y
        print center_x

        if w > h:
            r = int(0.64 * w)
        else:
            r = int(0.64 * h)
        new_x = center_x - r
        new_y = center_y - r
        print r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        cv2.imwrite(os.path.join( args.root_path, 'base_roi.png'),roi)
        
# get_base_roi(os.path.join(args.root_path, 'base.jpg'))


def get_base(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(os.path.join(args.root_path,'shape_predictor_68_face_landmarks.dat'))

    # try:
    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(image_path)
    # print type(image)
    # print image.shape
    # print image.shape[:-1]

    # new_im= cv2.resize(image, (int(image.shape[0] * 0.577),int(image.shape[0] * 0.577)), interpolation = cv2.INTER_AREA)
    # cv2.imwrite(image_path,new_im)
    # print new_im.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # print shape
        # print shape.shape
        print shape[37] - shape[41]
        shape[37:39,:] = shape[37:39,:] + np.array([0,4])
        shape[40:42,:] = shape[40:42,:] - np.array([0,1])
        shape[43:45,:] = shape[43:45,:] + np.array([0,4])
        shape[46:48,:] = shape[46:48,:] - np.array([0,1])

        nose_eye = shape[17:48]
        np.save( os.path.join( args.root_path, 'base_68_close.npy'),shape)
        # try:
        land2d = np.zeros((1,128,128),dtype=int)
        for inx in range(len(nose_eye)):
            try:
                x=int(nose_eye[inx][1])
                y =int(nose_eye[inx][0])
                print x,y
                land2d[0][x][y] = 255
                cv2.circle(image, (y, x), 1, (0, 0, 255), -1)
            except:
                continue

        land2d_p =  os.path.join(args.root_path,'base_show_close.png')
        cv2.imwrite(land2d_p,image)

        land2d_p =  os.path.join( args.root_path, 'base_2d_close.npy')
        np.save( land2d_p, land2d)
        print land2d_p
    # detector = dlib.get_frontal_face_detector()
    # predictor = dlib.shape_predictor(os.path.join(args.root_path,'shape_predictor_68_face_landmarks.dat'))

    # # try:
    # # load the input image, resize it, and convert it to grayscale
    # image = cv2.imread(image_path)
    # # print type(image)
    # # print image.shape
    # # print image.shape[:-1]

    # # new_im= cv2.resize(image, (int(image.shape[0] * 0.577),int(image.shape[0] * 0.577)), interpolation = cv2.INTER_AREA)
    # # cv2.imwrite(image_path,new_im)
    # # print new_im.shape
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # detect faces in the grayscale image
    # rects = detector(gray, 1)
    # for (i, rect) in enumerate(rects):

    #     shape = predictor(gray, rect)
    #     shape = face_utils.shape_to_np(shape)
    #     print shape
    #     print shape.shape

    #     nose_eye = shape[17:48]
    #     np.save( os.path.join( args.root_path, 'base_68.npy'),shape)
    #     # try:
    #     land2d = np.zeros((1,int(image.shape[0] * 0.577),int(image.shape[0] * 0.577)),dtype=int)
    #     for inx in range(len(nose_eye)):
    #         x=int(nose_eye[inx][1])
    #         y =int(nose_eye[inx][0])
    #         land2d[0][x][y] = 255
    #         cv2.circle(image, (y, x), 1, (0, 0, 255), -1)

    #     land2d_p =  os.path.join(args.root_path,'base_show.png')
    #     cv2.imwrite(land2d_p,new_im)

    #     land2d_p =  os.path.join( args.root_path, 'base_2d.npy')
    #     np.save( land2d_p, land2d)
    #     print land2d_p

# get_base(os.path.join(args.root_path, 'base_roi.jpg'))
    
      
def crop_image(image_path):
    

  
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

 
    rects = detector(gray, 1)
    for (i, rect) in enumerate(rects):

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
      
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        center_x = x + int(0.5 * w)
        center_y = y + int(0.5 * h)

        
        r = int(0.64 * h)
        new_x = center_x - r
        new_y = center_y - r
        roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]
        
        roi = cv2.resize(roi, (163,163), interpolation = cv2.INTER_AREA)
        scale =  163. / (2 * r)
       
        shape = ((shape - np.array([new_x,new_y])) * scale)
    
        return roi, shape 



def generating_landmark_lips(lists):
    global CC
    image_txt = lists

    
    for line in image_txt:
        img_path = line
        temp = img_path.split('/')
       

        if not os.path.exists(os.path.join(args.root_path,'regions') + '/' + temp[-4]):
            os.mkdir(os.path.join(args.root_path,'regions') + '/' + temp[-4])

        if not os.path.exists(os.path.join(args.root_path,'landmark1d') + '/' + temp[-4]):
            os.mkdir(os.path.join(args.root_path,'landmark1d') + '/' + temp[-4])

        if not os.path.exists(os.path.join(args.root_path,'regions') + '/' + temp[-4] + '/' + temp[-3]):
            os.mkdir(os.path.join(args.root_path,'regions') + '/' + temp[-4] + '/' + temp[-3])

        if not os.path.exists(os.path.join(args.root_path,'landmark1d') + '/' + temp[-4] + '/' + temp[-3]):
            os.mkdir(os.path.join(args.root_path,'landmark1d') + '/' + temp[-4] + '/' + temp[-3])

        if not os.path.exists(
                os.path.join(args.root_path,'regions') + '/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2]):
            os.mkdir(os.path.join(args.root_path,'regions') + '/' + temp[-4] +
                     '/' + temp[-3] + '/' + temp[-2])

        if not os.path.exists(os.path.join(args.root_path,'landmark1d') + '/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2]):
            os.mkdir(os.path.join(args.root_path,'landmark1d') + '/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2])


        landmark_path = os.path.join(args.root_path,'landmark1d') + '/' + \
            temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.npy'
        lip_path = os.path.join(args.root_path,'regions') + '/' + \
            temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.jpg'
        # if os.path.exists(landmark_path):
        #     continue
        try:
        
            roi, landmark= crop_image(img_path)
            if  np.sum(landmark[37:39,1] - landmark[40:42,1]) < -9:

                # pts2 = np.float32(np.array([template[36],template[45],template[30]]))
                template = np.load(os.path.join(args.root_path, 'base_68.npy'))
            else:
                template = np.load(os.path.join(args.root_path, 'base_68_close.npy'))
            # pts2 = np.float32(np.vstack((template[27:36,:], template[39,:],template[42,:],template[45,:])))
            pts2 = np.float32(template[27:45,:])
            # pts2 = np.float32(template[17:35,:])
            # pts1 = np.vstack((landmark[27:36,:], landmark[39,:],landmark[42,:],landmark[45,:]))
            pts1 = np.float32(landmark[27:45,:])
            # pts1 = np.float32(landmark[17:35,:])
            tform = tf.SimilarityTransform()
            tform.estimate( pts2, pts1)
            dst = tf.warp(roi, tform, output_shape=(163, 163))

            dst = np.array(dst * 255, dtype=np.uint8)
            dst = dst[1:129,1:129,:]
            cv2.imwrite(lip_path, dst)
        

            gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

            # detect faces in the grayscale image
            rects = detector(gray, 1)
            for (i, rect) in enumerate(rects):

                shape = predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                (x, y, w, h) = cv2.boundingRect(shape[48:,:])

                center_x = x + int(0.5 * w)

                center_y = y + int(0.5 * h)

                if w > h:
                    r = int(0.65 * w)
                else:
                    r = int(0.65 * h)
                new_x = center_x - r
                new_y = center_y - r
                roi = dst[new_y:new_y + 2 * r, new_x:new_x + 2 * r]

                cv2.imwrite(lip_path.replace('.jpg','#lip.jpg'), roi)


                for inx in range(len(shape)):
                    x=int(shape[inx][1])
                    y =int(shape[inx][0])
                    cv2.circle(dst, (y, x), 1, (0, 0, 255), -1)
                cv2.imwrite(landmark_path.replace('.npy','.jpg'), dst)
                np.save(landmark_path, shape)
            print CC 
            CC += 1
            
            
        except:
            continue
            print line



def multi_pool():
    if not os.path.exists(os.path.join(args.root_path,'regions') ):
        os.mkdir(os.path.join(args.root_path,'regions') )
    if not os.path.exists(os.path.join(args.root_path,'landmark1d') ):
        os.mkdir(os.path.join(args.root_path,'landmark1d') )
    data = []
    words =[]
    for path in os.listdir(os.path.join(args.root_path,'image')):
        words.append(path)
    words = sorted(words)
    print len(words)
    # words = [words[0]]
    words = words[: int(0.1 * len(words))]
    for w in words:

        for path, subdirs, files in os.walk(os.path.join(os.path.join(args.root_path,'image'), w) ):
            for name in files:
                data.append( os.path.join(path, name))

    num_thread = 25
    datas = []
    batch_size = int(len(data) / num_thread)
    temp = []
    for i, d in enumerate(data):
        temp.append(d)
        if (i + 1) % batch_size == 0:
            datas.append(temp)
            temp = []
    for i in range(num_thread):
        process = multiprocessing.Process(
            target=generating_landmark_lips, args=(datas[i],))
        process.start()


# multi_pool()



def generate_lms():
    txt = open(path + '/prefix.txt', 'r')
    image_txt = open(path + '/image.txt', 'w')
    if not os.path.exists(path + '/lms'):
        os.mkdir(path + '/lms')
    if not os.path.exists(path + '/chunk'):
        os.mkdir(path + '/chunk')

    count = 0
    for line in txt:
        line = line[:-1].replace('lipread_vgg', 'lrw')
        temp = line.split('/')
        word = temp[-3]
        mode = temp[-2]
        video = temp[-1][:-4]
        if not os.path.exists(path + '/lms/' + word):
            os.mkdir(path + '/lms/' + word)
        if not os.path.exists(path + '/lms/' + word + '/' + mode):
            os.mkdir(path + '/lms/' + word + '/' + mode)
        if not os.path.exists(
            path +
            '/lms/' +
            word +
            '/' +
            mode +
            '/' +
                video):
            os.mkdir(path + '/lms/' + word + '/' + mode + '/' + video)
        if not os.path.exists(path + '/chunk/' + word):
            os.mkdir(path + '/chunk/' + word)
        if not os.path.exists(path + '/chunk/' + word + '/' + mode):
            os.mkdir(path + '/chunk/' + word + '/' + mode)
        if not os.path.exists(
            path +
            '/chunk/' +
            word +
            '/' +
            mode +
            '/' +
                video):
            os.mkdir(path + '/chunk/' + word + '/' + mode + '/' + video)

        image_path = path + '/image/' + word + '/' + mode + '/' + video
        imgs = os.listdir(image_path)
        frame_num = len(imgs)
        audio_path = path + '/audio/' + word + '/' + mode + '/' + video + '.wav'
        fs, y = scipy.io.wavfile.read(audio_path)
        chunk_len = len(y) * 1.0 / frame_num
        for i, img_name in enumerate(imgs):
            audio_start_frame = int(i * chunk_len)
            audio_end_frame = int((i + 1) * chunk_len)
            chunk = y[audio_start_frame:audio_end_frame]

            lms = wav2lms(chunk, fs)
            chunk_name = path + '/chunk/' + word + '/' + \
                mode + '/' + video + '/' + img_name[:-4] + '.npy'
            lms_name = path + '/lms/' + word + '/' + mode + \
                '/' + video + '/' + img_name[:-4] + '.npy'
            if lms.shape != (128, 4):
                print audio_path
                print '++'
                print lms.shape
                break
            print str(count)
            count += 1
            np.save(lms_name, lms)
            np.save(chunk_name, chunk)
            image_txt.write(
                path +
                '/image/' +
                word +
                '/' +
                mode +
                '/' +
                video +
                '/' +
                img_name +
                '\n')


  

def wav2lms(wav=None, sr=44100):
    y = wav

    S = librosa.feature.melspectrogram(
        y, sr=sr, n_fft=1024, n_mels=128, fmax=16000)
    log_S = librosa.logamplitude(S)
    return log_S


def get_data():

    train_set = []
    test_set = []
    count = 0

    for root, dirs, files in os.walk(os.path.join(args.root_path,'landmark1d') ):
        # if count == 5:
        #     break
        line = root
        if len(line.split('/'))!= 11:
            continue
        image_path = line
        if '/train/' in image_path:
            train_set.append(image_path)
        elif '/test/' in image_path:
            test_set.append(image_path)
        elif '/val/' in image_path:
            train_set.append(image_path)
        else:
            print 'wrong'
        if len(os.listdir(root)) == 29 * 2:
            count += 1
    # print len(train_set)
    # print len(test_set)
    # print test_set[0]
    print count
    return [train_set, test_set]


def generate_video_pickle(datalists=None):
    data = []
    for dataset in datalists:
        temp = []
       
        for i in xrange(0, len(dataset)):
            video_id = dataset[i].split('/')[-1]
            fff = {}
            print dataset[i]
            fff["image_path"] = []
            fff["lms_path"] = []
            fff['landmark_path'] = []

            for  current_frame_id in range(1,30):
                landmark_frame = os.path.join(dataset[i],video_id+'_%03d.npy'%current_frame_id)
                # print video_frame
                if os.path.isfile(landmark_frame.replace('landmark2d','lips').replace('.npy','.jpg')):
                    fff["image_path"].append(landmark_frame.replace('landmark2d','lips').replace('.npy','.jpg'))
                if os.path.isfile(landmark_frame.replace('landmark2d','lms')):
                    fff["lms_path"].append(landmark_frame.replace('landmark2d','lms'))
                if os.path.isfile(landmark_frame):
                    if np.load(landmark_frame).shape == (1,64,64):
                        fff["landmark_path"].append(landmark_frame)
                if current_frame_id == 29 and len(fff['image_path'])==29 and len(fff['landmark_path']) == 29:
                    temp.append(fff)
            


                    

        data.append(temp)

    print len(data)
    print len(data[0])
    print len(data[1])
    with open(os.path.join(args.root_path,'pickle/train.pkl'), 'wb') as handle:
        pickle.dump(data[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.root_path,'pickle/test.pkl'), 'wb') as handle:
        pickle.dump(data[1], handle, protocol=pickle.HIGHEST_PROTOCOL)

def generate_image_pickle(datalists=None):
    data = []
    for dataset in datalists:
        temp = []
       
        for i in xrange(0, len(dataset)):
           
            video_id = dataset[i].split('/')[-1]
            print '{}/{}'.format(i,len(dataset))
          

            for  current_frame_id in range(1,30):
                flage = True
                fff = []
                landmark_frame = os.path.join(dataset[i],video_id+'_%03d.npy'%current_frame_id)
                if os.path.isfile(landmark_frame.replace('landmark1d','regions').replace('.npy','.jpg')):
                    fff.append(landmark_frame.replace('landmark1d','regions').replace('.npy','.jpg'))
                else:
                    flage = False
                if os.path.isfile(landmark_frame.replace('landmark1d','lms')):
                    fff.append(landmark_frame.replace('landmark1d','lms'))
                else:
                    flage = False
                if os.path.isfile(landmark_frame):
                    fff.append(landmark_frame)
                else:
                    flage = False
                if flage:
                    temp.append(fff)
            

                    
        data.append(temp)
    # print data[0][0:3]
    # print len(data[0])
    # print len(data[1])
    with open(os.path.join(args.root_path,'pickle/new_img_train.pkl'), 'wb') as handle:
        pickle.dump(data[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.root_path,'pickle/new_img_test.pkl'), 'wb') as handle:
        pickle.dump(data[1], handle, protocol=pickle.HIGHEST_PROTOCOL)




def generate_video_pickle(datalists=None):
    data = []
    for dataset in datalists:
        temp = []
       
        for i in xrange(0, len(dataset)):
           
            video_id = dataset[i].split('/')[-1]
            print '{}/{}'.format(i,len(dataset))
            flage = True

            fff = []
            for  current_frame_id in range(1,30):
                
                landmark_frame = os.path.join(dataset[i],video_id+'_%03d.npy'%current_frame_id)
                if os.path.isfile(landmark_frame.replace('landmark1d','regions').replace('.npy','.jpg')):
                    fff.append(landmark_frame.replace('landmark1d','regions').replace('.npy','.jpg'))
                else:
                    flage = False
                if os.path.isfile(landmark_frame.replace('landmark1d','lms')):
                    fff.append(landmark_frame.replace('landmark1d','lms'))
                else:
                    flage = False
                if os.path.isfile(landmark_frame):
                    fff.append(landmark_frame)
                else:
                    flage = False
            if flage:
                temp.append(fff)
            

        print len(temp)
        data.append(temp)
    # print data[0][0:3]
    # print len(data[0])
    # print len(data[1])
    with open(os.path.join(args.root_path,'pickle/new_video_train.pkl'), 'wb') as handle:
        pickle.dump(data[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.root_path,'pickle/new_video_test.pkl'), 'wb') as handle:
        pickle.dump(data[1], handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_video_16_pickle(datalists=None):
    data = []
    for dataset in datalists:
        temp = []
       
        for i in xrange(0, len(dataset)):
           
            video_id = dataset[i].split('/')[-1]
            print '{}/{}'.format(i,len(dataset))
            flage = True

            fff = []

            for  current_frame_id in xrange(1,30,8):
                fff= []

                for inx in range(16):
                
                    landmark_frame = os.path.join(dataset[i],video_id + '_%03d.npy'% (current_frame_id + inx))
                    if os.path.isfile(landmark_frame.replace('landmark1d','regions').replace('.npy','.jpg')):
                        fff.append(landmark_frame.replace('landmark1d','regions').replace('.npy','.jpg'))
                    else:
                        flage = False
                    if os.path.isfile(landmark_frame.replace('landmark1d','lms')):
                        fff.append(landmark_frame.replace('landmark1d','lms'))
                    else:
                        flage = False
                    if os.path.isfile(landmark_frame):
                        fff.append(landmark_frame)
                    else:
                        flage = False
                print len(fff)
                if flage == True and len(fff) == 48:
                    temp.append(fff)
                    print '+++'
            

        print len(temp)
        data.append(temp)
    # print data[0][0:3]
    # print len(data[0])
    # print len(data[1])
    with open(os.path.join(args.root_path,'pickle/new_video_16_train.pkl'), 'wb') as handle:
        pickle.dump(data[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.root_path,'pickle/new_video_16_test.pkl'), 'wb') as handle:
        pickle.dump(data[1], handle, protocol=pickle.HIGHEST_PROTOCOL)


# generate_txt()
# extract_audio()
# extract_images()
# get_base(os.path.join(args.root_path, 'base_roi.jpg'))
# generate_lms()
# multi_pool()
# multi_pool_2dland()
datalists = get_data()
# print datalists
# generate_image_pickle(datalists)
generate_video_16_pickle(datalists)

# with open('/home/lele/Music/text-to-image.pytorch/data/train.pkl', 'rb') as handle:
#     b = pickle.load(handle)
# print b[0]

