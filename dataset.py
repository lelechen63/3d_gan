import os
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class LRWdataset1D_3d(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[128, 128],
                 train='train'):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = tuple(output_shape)

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train=='train':
            _file = open(os.path.join(dataset_dir, "new_video_8_train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        elif self.train =='test':
            _file = open(os.path.join(dataset_dir, "new_video_8_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()
        elif self.train =='demo' :
            _file = open(os.path.join(dataset_dir, "img_demo.pkl"), "rb")
            self.demo_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train=='train':
        
            while True:
                # try:
                    wrong_index = random.choice(
                        [x for x in range(self.__len__()) if x != index])
                    #load righ img
                    right_imgs = torch.FloatTensor(16,128,128,3)
                    right_lmss = torch.FloatTensor(1,64,128)
                    # right_landmarks = torch.FloatTensor(8,68,2)

                    wrong_imgs = torch.FloatTensor(16,128,128,3)
                    wrong_lmss = torch.FloatTensor(1, 64,128)
                    # wrong_landmark = torch.FloatTensor(8,68,2)

                    for i in  range(0,16):
                        print index
                        print i*3
                        print len(self.train_data)
                        print len(self.train_data[index])
                        image_path = self.train_data[index][i*3]
                        lms_path = self.train_data[index][1 + i*3]
                        # landmark_path = self.train_data[index][2 + i*3]
                        im = cv2.imread(image_path)
                        if im is None:
                            raise IOError
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = cv2.resize(im, self.output_shape)
                        im = self.transform(im)
                        right_img = torch.FloatTensor(im)
                        # right_landmark = torch.FloatTensor(np.load(landmark_path))
                        right_lms = torch.FloatTensor(np.load(lms_path))
                        right_imgs[i,:,:,:] = right_img
                        # right_landmarks[i,:,:] =  right_landmark
                        right_lmss[0,i*4:(i+1) * 4,:] =  right_lms


                        image_path = self.train_data[wrong_index][i*3]
                        lms_path = self.train_data[wrong_index][1 + i*3]
                        # landmark_path = self.train_data[wrong_index][2 + i*3]
                        im = cv2.imread(image_path)
                        if im is None:
                            raise IOError
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = cv2.resize(im, self.output_shape)
                        im = self.transform(im)
                        wrong_img = torch.FloatTensor(im)
                        # wrong_landmark = torch.FloatTensor(np.load(landmark_path))
                        wrong_lms = torch.FloatTensor(np.load(lms_path))
                        wrong_imgs[i,:,:,:] = wrong_img
                        # wrong_landmarks[i,:,:,:] =  wrong_landmark
                        wrong_lmss[0,i*4:(i+1) * 4,:] =  wrong_lms

                    example_image = right_imgs[0]
                    # example_landmark = right_landmarks[0]
                    example_lms = right_lmss[0]

                    # return  example_landmark, example_lms,right_landmarks, right_lmss
                  
                    # return example_image, example_landmark, example_lms, right_imgs,right_landmarks, right_lmss
                    right_imgs = right_imgs.permute(3,0,1,2)
                    wrong_imgs = wrong_imgs.permute(3,0,1,2)
                    example_image = example_image.permute(2,0,1)

                    return example_image, example_lms, right_imgs, right_lmss, wrong_imgs,wrong_lmss

                    # return example_image, example_landmark, example_lms, right_imgs,right_landmarks, right_lmss, wrong_imgs, wrong_landmarks,wrong_lmss
                # except:
                #     index = (index + 1) % len(self.train_data)

                    # print 'Fuck'

        elif self.train =='test':
            while True:
                # try:
                    wrong_index = random.choice(
                        [x for x in range(self.__len__()) if x != index])
                    #load righ img
                    right_imgs = torch.FloatTensor(16,128,128,3)
                    right_lmss = torch.FloatTensor(1,64,128)
                    # right_landmarks = torch.FloatTensor(8,68,2)

                    wrong_imgs = torch.FloatTensor(16,128,128,3)
                    wrong_lmss = torch.FloatTensor(1,64,128)
                    # wrong_landmark = torch.FloatTensor(8,68,2)

                    for i in  range(0,16):
                        image_path = self.train_data[index][i*3]
                        lms_path = self.test_data[index][1 + i*3]
                        landmark_path = self.test_data[index][2 + i*3]
                        im = cv2.imread(image_path)
                        if im is None:
                            raise IOError
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = cv2.resize(im, self.output_shape)
                        im = self.transform(im)
                        right_img = torch.FloatTensor(im)
                        # right_landmark = torch.FloatTensor(np.load(landmark_path))
                        right_lms = torch.FloatTensor(np.load(lms_path))
                        right_imgs[i,:,:,:] = right_img
                        # right_landmarks[i,:,:] =  right_landmark
                        right_lmss[0,i*4:(i+1) * 4,:] =  right_lms


                        image_path = self.train_data[wrong_index][i*3]
                        lms_path = self.train_data[wrong_index][1 + i*3]
                        # landmark_path = self.train_data[wrong_index][2 + i*3]
                        im = cv2.imread(image_path)
                        if im is None:
                            raise IOError
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        im = cv2.resize(im, self.output_shape)
                        im = self.transform(im)
                        wrong_img = torch.FloatTensor(im)
                        # wrong_landmark = torch.FloatTensor(np.load(landmark_path))
                        wrong_lms = torch.FloatTensor(np.load(lms_path))
                        wrong_imgs[i,:,:,:] = wrong_img
                        # wrong_landmarks[i,:,:,:] =  wrong_landmark
                        wrong_lmss[0,i*4:(i+1) * 4,:] =  wrong_lms

                    example_image = right_imgs[0]
                    # example_landmark = right_landmarks[0]
                    example_lms = right_lmss[0]
                    
                    # return  example_landmark, example_lms,right_landmarks, right_lmss
                    # return example_image, example_landmark, example_lms, right_imgs,right_landmarks, right_lmss
                    right_imgs = right_imgs.permute(3,0,1,2)
                    wrong_imgs = wrong_imgs.permute(3,0,1,2)

                    example_image = example_image.permute(2,0,1)
                    return example_image, example_lms, right_imgs, right_lmss, wrong_imgs, wrong_lmss
                   

                    # return example_image, example_landmark, example_lms, right_imgs,right_landmarks, right_lmss, wrong_imgs, wrong_landmarks,wrong_lmss
                # except:
                #     index = (index + 1) % len(self.train_data)
        elif  self.train =='demo':
            while True:
                # try:
                
                    image_path = self.demo_data[index][0]
                    landmark_path = self.demo_data[index][2]
                    im = cv2.imread(image_path)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                    im = cv2.resize(im, self.output_shape)
                    im = self.transform(im)

                    right_img = torch.FloatTensor(im)


                    #load example image
                    
                    example_path = image_path.split('_')[0] + '_00001.jpg'
                    example_landmark = np.load(landmark_path.split('_')[0] + '_00001.npy')

                    # example_path = '/mnt/disk1/dat/lchen63/lrw/demo/obama_cartoon_region.jpg'
                    # example_landmark = np.load('/mnt/disk1/dat/lchen63/lrw/demo/obama_cartoon.npy')

                    example_lip = cv2.imread(example_path)
                 
                    example_lip = cv2.cvtColor(example_lip, cv2.COLOR_BGR2RGB)
                    example_lip = cv2.resize(example_lip, self.output_shape)
                    example_lip = self.transform(example_lip)


                    land_path = self.demo_data[index][2]
                    right_landmark = torch.FloatTensor(np.load(landmark_path))
      
                    wrong_landmark = right_landmark
                    return example_lip, example_landmark, right_img,right_landmark, wrong_landmark
                    # except:
                    #     print '#############'
                    #     ndex = (index + 1) % len(self.demo_data)
                    # continue
                    # print 'Fuck'

    def __len__(self):
        if self.train=='train':
            return len(self.train_data)
        elif self.train=='test':
            return len(self.test_data)
        else:
            return len(self.demo_data)
