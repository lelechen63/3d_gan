import pickle
import os
import scipy.io.wavfile
import librosa
import numpy as np
import argparse
from imutils import face_utils
import dlib
import cv2
import multiprocessing


def parse_arguments():
    """Parse arguments from command line"""
    description = "Train a model."
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--root_path', '-p',
                        default="/mnt/disk1/dat/lchen63/lrw/data",
                        help='data path'
                        )
    parser.add_argument(
        '--shape_predictor',
        '-sp',
        default="/home/lchen63/project/text-to-image.pytorch/data/shape_predictor_68_face_landmarks.dat",
        help='data path')

    return parser.parse_args()


args = parse_arguments()
path = args.root_path

global CC
CC = 0


def generate_txt():
    txt = open(path + '/prefix.txt', 'w')
    for root, dirs, files in os.walk(path + '/video/'):
        for file in files:
            if file[-3:] == 'mov':
                name = os.path.join(root, file)
                txt.write(name + '\n')


def _extract_images(lists):
    global CC
    for line in lists:
        CC += 1
        print '++++++++++++++++++++++++++++++++++++++++++++++++++++' + str(CC) + '/' + str(len(lists))
        temp = line.split('/')
        if not os.path.exists(path + '/image/' + temp[-2]):
            os.mkdir(path + '/image/' + temp[-2])

        if not os.path.exists(
                path + '/image/' + temp[-2] + '/' + temp[-1][:-4]):
            os.mkdir(path + '/image/' + temp[-2] + '/' + temp[-1][:-4])

        command = ' ffmpeg -i ' + line + ' -r 25 ' + path + '/image/' + \
            temp[-2] + '/' + temp[-1][:-4] + '/' + temp[-1][:-4] + '_%03d.jpg'
        print command
        try:
            # pass
            os.system(command)
        except BaseException:
            print line
    # else:
    #     continue


def extract_images():
    txt = open(path + '/prefix.txt', 'r')
    if not os.path.exists(path + '/image/'):
        os.mkdir(path + '/image/')
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
    global CC
    CC = 0
    for line in lists:
        CC += 1
        print '++++++++++++++++++++++++++++++++++++++++++++++++++++' + str(CC) + '/' + str(len(lists))
        temp = line.split('/')
        if not os.path.exists(path + '/audio/' + temp[-3]):
            os.mkdir(path + '/audio/' + temp[-3])
        if not os.path.exists(path + '/audio/' + temp[-3] + '/' + temp[-2]):
            os.mkdir(path + '/audio/' + temp[-3] + '/' + temp[-2])

        command = 'ffmpeg -i ' + line + ' -ar 44100  -ac 1  ' + path + \
            '/audio/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.wav'
        print command
        try:
            os.system(command)
        except BaseException:
            print line


def extract_audio():
    txt = open(path + '/prefix.txt', 'r')
    if not os.path.exists(path + '/audio/'):
        os.mkdir(path + '/audio/')
    total = []
    for line in txt:
        total.append(line[:-1].replace('lipread_vgg', 'lrw'))
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


def crop_lips(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.shape_predictor)

    try:
        # load the input image, resize it, and convert it to grayscale
        image = cv2.imread(image_path)
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

                center_x = x + int(0.5 * w)

                center_y = y + int(0.5 * h)

                if w > h:
                    r = int(0.65 * w)
                else:
                    r = int(0.65 * h)
                new_x = center_x - r
                new_y = center_y - r
                roi = image[new_y:new_y + 2 * r, new_x:new_x + 2 * r]

                return shape[i:j], roi
    except BaseException:
        print image_path


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


def generating_landmark_lips(lists):
    # image_txt = open(path + 'image.txt','r')
    image_txt = lists
    for line in image_txt:
        img_path = line
        temp = img_path.split('/')
        if os.path.isfile(path + '/landmark/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.npy') and os.path.isfile(
                path + '/lips/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.jpg'):
            print '---'
            print temp[-1]
            continue

        if not os.path.exists(path + '/lips/' + temp[-4]):
            os.mkdir(path + '/lips/' + temp[-4])

        if not os.path.exists(path + '/landmark/' + temp[-4]):
            os.mkdir(path + '/landmark/' + temp[-4])

        if not os.path.exists(path + '/lips/' + temp[-4] + '/' + temp[-3]):
            os.mkdir(path + '/lips/' + temp[-4] + '/' + temp[-3])

        if not os.path.exists(path + '/landmark/' + temp[-4] + '/' + temp[-3]):
            os.mkdir(path + '/landmark/' + temp[-4] + '/' + temp[-3])

        if not os.path.exists(
                path + '/lips/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2]):
            os.mkdir(path + '/lips/' + temp[-4] +
                     '/' + temp[-3] + '/' + temp[-2])

        if not os.path.exists(path + '/landmark/' +
                              temp[-4] + '/' + temp[-3] + '/' + temp[-2]):
            os.mkdir(path + '/landmark/' +
                     temp[-4] + '/' + temp[-3] + '/' + temp[-2])

        # if not os.path.exists(path + 'lips/' +temp[-4] + '/' + temp[-3] + '/' + temp[-2]+ '/' + temp[-1] ):
        #     os.mkdir(path + 'lips/' + temp[-4] + '/' + temp[-3] + '/' + temp[-2]+ '/' + temp[-1])

        # if not os.path.exists(path + 'landmark/' +temp[-4] + '/' + temp[-3]+ '/' + temp[-2]+ '/' + temp[-1]):
        #     os.mkdir(path + 'landmark/' + temp[-4] + '/' + temp[-3]+ '/' + temp[-2]+ '/' + temp[-1])

        landmark_path = path + '/landmark/' + \
            temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.npy'
        lip_path = path + '/lips/' + \
            temp[-4] + '/' + temp[-3] + '/' + temp[-2] + '/' + temp[-1][:-4] + '.jpg'
        try:
            landmark, lip = crop_lips(img_path)
            print lip.shape
            print landmark.shape
            cv2.imwrite(lip_path, lip)
            np.save(landmark_path, landmark)

        except BaseException:
            print line


def multi_pool():
    data = []
    image_txt = open(path + '/image.txt', 'r')
    for line in image_txt:
        data.append(line[:-1])
    num_thread = 30
    datas = []
    batch_size = int(len(data) / num_thread)
    temp = []
    for i, d in enumerate(data):
        temp.append(d)
        if (i + 1) % batch_size == 0:
            datas.append(temp)
            temp = []
    print len(datas)
    for i in range(num_thread):
        process = multiprocessing.Process(
            target=generating_landmark_lips, args=(datas[i],))
        process.start()


def wav2lms(wav=None, sr=44100):
    y = wav

    S = librosa.feature.melspectrogram(
        y, sr=sr, n_fft=1024, n_mels=128, fmax=16000)
    log_S = librosa.logamplitude(S)
    return log_S


def get_data():
    data_txt = open(path + '/image.txt')

    # count = 0
    train_set = []
    test_set = []
    for line in data_txt:
        # print line
        # count += 1
        # if count == 200:
        #     break
        image_path = line[:-1]
        if '/train/' in image_path:
            train_set.append(image_path)
        elif '/test/' in image_path:
            test_set.append(image_path)
        elif '/val/' in image_path:
            train_set.append(image_path)
        else:
            print 'wrong!'
    train_set = sorted(train_set)
    test_set = sorted(test_set)

    print len(train_set)
    print len(test_set)
    return [train_set, test_set]


def generate_video_pickle(datalists=None):
    data = []
    for dataset in datalists:
        temp = []
        for i in xrange(0, len(dataset), 8):
            fff = {}
            start_fram = int(dataset[i].split('_')[-1].split('.')[0])

            img_path = dataset[i].replace('image', 'lips')[
                :-len(dataset[i].split('/')[-1])]
            lms_path = dataset[i].replace('image', 'lms')[
                :-len(dataset[i].split('/')[-1])]

            imgs = []
            lmss = []
            for j in range(0, 16):
                img_f = img_path + \
                    dataset[i].split('/')[-2] + '_%03d.jpg' % (j + start_fram)
                lms_f = lms_path + \
                    dataset[i].split('/')[-2] + '_%03d.npy' % (j + start_fram)
                if os.path.isfile(img_f) and os.path.isfile(lms_f):
                    imgs.append(img_f)
                    lmss.append(lms_f)
                else:
                    # count += 1
                    print img_f
                    break
                fff["image_path"] = imgs
                fff["lms_path"] = lmss
                if j == 15:
                    print '-----{}/{}'.format(len(temp), len(dataset))
                    temp.append(fff)
        data.append(temp)
    print len(data[0])
    print len(data[1])
    with open('/mnt/disk1/dat/lchen63/lrw/data/pickle/train.pkl', 'wb') as handle:
        pickle.dump(data[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('/mnt/disk1/dat/lchen63/lrw/data/pickle/test.pkl', 'wb') as handle:
        pickle.dump(data[1], handle, protocol=pickle.HIGHEST_PROTOCOL)


# train_set = []
# # count = 0
# def mmp_train(datalist):
#     dataset = datalist
#     temp = []
#     for i in xrange(0,len(dataset),4):
#         fff = {}
#         start_fram = int(dataset[i].split('_')[-1].split('.')[0])

#         img_path =dataset[i].replace('image','lips')[:-len(dataset[i].split('/')[-1])]
#         lms_path = dataset[i].replace('image','lms')[:-len(dataset[i].split('/')[-1])]

#         imgs = []
#         lmss = []
#         for j in range(0,16):
#             img_f = img_path +  dataset[i].split('/')[-2] + '_%03d.jpg'%(j + start_fram)
#             lms_f = lms_path +  dataset[i].split('/')[-2] + '_%03d.npy'%(j + start_fram)
#             if  os.path.isfile(img_f) and os.path.isfile(lms_f):
#                     imgs.append(img_f)
#                     lmss.append(lms_f)
#             else:
#                 # count += 1
#                 print img_f
#                 break
#             fff["image_path"]= imgs
#             fff["lms_path"]= lmss
#             if j == 15:
#                 print '-----'+ str(len(train_set))
#                 train_set.append(fff)

#     # train_set.extend(temp)
# def mmp_test(datalist):
#     dataset = datalist
#     temp = []
#     for i in xrange(0,len(dataset),8):
#         fff = {}
#         start_fram = int(dataset[i].split('_')[-1].split('.')[0])

#         img_path =dataset[i].replace('image','lips')[:-len(dataset[i].split('/')[-1])]
#         lms_path = dataset[i].replace('image','lms')[:-len(dataset[i].split('/')[-1])]

#         imgs = []
#         lmss = []
#         for j in range(0,16):
#             img_f = img_path +  dataset[i].split('/')[-2] + '_%03d.jpg'%(j + start_fram)
#             lms_f = lms_path +  dataset[i].split('/')[-2] + '_%03d.npy'%(j + start_fram)
#             if  os.path.isfile(img_f) and os.path.isfile(lms_f):
#                     imgs.append(img_f)
#                     lmss.append(lms_f)
#             else:
#                 # count += 1
#                 print img_f
#                 break
#             fff["image_path"]= imgs
#             fff["lms_path"]= lmss
#             if j == 15:
#                 print '-----'
#                 temp.append(fff)
#     test_set.extend(temp)
# def generate_train_video_pickle(datalists = None):
#     count = 0
#     data = datalists[0]
#     num_thread = 30

#     datas = []
#     batch_size = int(len(data)/num_thread)
#     temp = []
#     for  i,d in enumerate (data):
#         temp.append(d)
#         if (i + 1) % batch_size == 0:
#             datas.append(temp)
#             temp = []
#     print len(datas)
#     for i in range(num_thread):
#         process = multiprocessing.Process(target = mmp_train,args = (datas[i],))
#         process.start()

    # for dataset in datalists:
    #     temp = []
    #     for i in xrange(0,len(dataset),8):
    #         fff = {}
    #         start_fram = int(dataset[i].split('_')[-1].split('.')[0])

    #         img_path =dataset[i].replace('image','lips')[:-len(dataset[i].split('/')[-1])]
    #         lms_path = dataset[i].replace('image','lms')[:-len(dataset[i].split('/')[-1])]

    #         imgs = []
    #         lmss = []
    #         for j in range(0,16):
    #             img_f = img_path +  dataset[i].split('/')[-2] + '_%03d.jpg'%(j + start_fram)
    #             lms_f = lms_path +  dataset[i].split('/')[-2] + '_%03d.npy'%(j + start_fram)
    #             if  os.path.isfile(img_f) and os.path.isfile(lms_f):
    #                     imgs.append(img_f)
    #                     lmss.append(lms_f)
    #             else:
    #                 count += 1
    #                 print img_f
    #                 break
    #             fff["image_path"]= imgs
    #             fff["lms_path"]= lmss
    #             if j == 15:
    #                 print '-----'
    #                 temp.append(fff)
    #     random.shuffle(temp)
    #     data.append(temp)
    # print len(data[0])
    # print len(data[1])
    # print 'training:\t'+ str(int(len(data)*0.9))
    # print 'testing:\t' + str(len(data) - int(len(data)*0.9))
    # train_data = data[:int(0.9*len(data))]
    # test_data = data[int(0.9*len(data)):]
    # random.shuffle(train_data)
    # random.shuffle(test_data)
    # random.shuffle(test)
    # with open('/mnt/disk1/dat/lchen63/lrw/data/pickle/train.pkl', 'wb') as handle:
    #     pickle.dump(train_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('/mnt/disk1/dat/lchen63/lrw/data/pickle/test.pkl', 'wb') as handle:
    #     pickle.dump(data[1], handle, protocol=pickle.HIGHEST_PROTOCOL)
    # with open('/mnt/disk1/dat/lchen63/lcd/data/pickle/new_test.pkl', 'wb') as handle:
    #     pickle.dump(test, handle, protocol=pickle.HIGHEST_PROTOCOL)



# generate_txt()
# extract_audio()
# extract_images()
# generate_lms()
# multi_pool()
datalists = get_data()
# print datalists
# generate_img_pickle(datalists)
generate_video_pickle(datalists)

# with open('/home/lele/Music/text-to-image.pytorch/data/train.pkl', 'rb') as handle:
#     b = pickle.load(handle)
# print b[0]

