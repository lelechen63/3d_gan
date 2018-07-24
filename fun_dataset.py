import os
import random
import pickle
import numpy as np
import cv2
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

class VaganDataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[64, 64],
                 train=True):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = output_shape

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train:
            _file = open(os.path.join(dataset_dir, "train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        else:
            _file = open(os.path.join(dataset_dir, "new_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text

        if self.train:
            paths =  self.train_data[index]["image_path"]

            # while True:
                # a = random.randint(0,75)
                # if paths[0][-8] == '_':
                #     example_path = paths[0][:-8] +'_%03d.jpg' %(a)
                #     if os.path.isfile(example_path):
                #         break
                # elif paths[0][-9] == '_':
                #     example_path = paths[0][:-9] +'_%03d.jpg' %(a)
                #     if os.path.isfile(example_path):
                #         break
            example_path = '/mnt/disk0/dat/lchen63/grid/test_fun/model_base/trump1_roi.jpg'
            example_lip = Image.open(example_path).convert("RGB").resize(self.output_shape)
            example_lip = self.transform(example_lip)
            # example_lip = example_lip.view(1,example_lip[0],example_lip[1],example_lip[2])

            im_cub = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
            for i,path in enumerate(paths):
                im = Image.open(path).convert("RGB").resize(self.output_shape)
                im = self.transform(im)
                im_cub[i,:,:,:] = im
            im_cub = im_cub.permute(1,0,2,3)
            wrong_index = random.choice(
                [x for x in range(self.__len__()) if x != index])

            right_lmss = self.train_data[index]["lms_path"]
            right_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(right_lmss):
                right_embed[0,i * 4: 4 *(i +1 ),:] = torch.FloatTensor(np.load(lms_path))

            # landmarks = torch.FloatTensor( 1, 16 , 20, 2)
            # for i,lms_path in enumerate(right_lmss):
            #     temp = np.load(lms_path.replace('lms','landmark'))[48:]
            #     original = np.sum(temp,axis=0) / 20.0
            #     temp = temp - original

            #     landmarks[0,i,:,:] = torch.FloatTensor(temp)
            landmarks = example_lip

            wrong_lmss = self.train_data[wrong_index]["lms_path"]
            wrong_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(wrong_lmss):
                wrong_embed[0,i * 4 : (i +1) * 4,:] = torch.FloatTensor(np.load(lms_path))

            return example_lip, im_cub, landmarks, right_embed, wrong_embed

        else:
            paths = self.test_data[index]["image_path"]

            # while True:
            #     a = random.randint(0,75)
            #     if paths[0][-8] == '_':
            #         example_path = paths[0][:-8] +'_%03d.jpg' %(a)
            #         if os.path.isfile(example_path):
            #             break
            #     elif paths[0][-9] == '_':
            #         example_path = paths[0][:-9] +'_%03d.jpg' %(a)
            #         if os.path.isfile(example_path):
            #             break
            example_path = '/mnt/disk0/dat/lchen63/grid/test_fun/model_base/trump3_roi.jpg'
            example_lip = Image.open(example_path).convert("RGB").resize(self.output_shape)
            example_lip = self.transform(example_lip)

            im_cub = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
            for i,path in enumerate(paths):
                im = Image.open(path).convert("RGB").resize(self.output_shape)
                im = self.transform(im)
                im_cub[i,:,:,:] = im
            im_cub = im_cub.permute(1,0,2,3)
            right_lmss = self.test_data[index]["lms_path"]
            right_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(right_lmss):
                right_embed[0,i * 4: 4 *(i +1 ),:] = torch.FloatTensor(np.load(lms_path))
            landmarks = torch.FloatTensor(1, 16 , 20,2)
            for i,lms_path in enumerate(right_lmss):

                temp = np.load(lms_path.replace('lms','landmark'))[48:]
                original = np.sum(temp,axis=0) / 20.0
                temp = temp - original

                landmarks[0,i,:,:] = torch.FloatTensor(temp)

            caption = self.test_data[index]["lms_path"]


            return example_lip,im_cub, landmarks, right_embed, caption

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class VaganFarnebackDataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[64, 64],
                 train=True):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = output_shape

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train:
            _file = open(os.path.join(dataset_dir, "train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        else:
            _file = open(os.path.join(dataset_dir, "new_test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.flow_transform = transforms.Normalize(mean=[0, 0], std=[20, 20])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text

        if self.train:
            paths =  self.train_data[index]["image_path"]

            while True:
                a = random.randint(0,75)
                if paths[0][-8] == '_':
                    example_path = paths[0][:-8] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
                elif paths[0][-9] == '_':
                    example_path = paths[0][:-9] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
            example_lip = Image.open(example_path).convert("RGB").resize(self.output_shape)
            example_lip = self.transform(example_lip)
            # example_lip = example_lip.view(1,example_lip[0],example_lip[1],example_lip[2])

            im_cub = torch.FloatTensor(16, 3, self.output_shape[0], self.output_shape[1])
            flow_cub = torch.FloatTensor(15, 2, self.output_shape[0], self.output_shape[1])

            prev_im = None
            for i, path in enumerate(paths):
                im = Image.open(path).convert("RGB").resize(self.output_shape)
                cur_im = im
                im = self.transform(im)
                im_cub[i] = im

                # estimate optical flow
                if i != 0:
                    prev_frm = cv2.cvtColor(np.array(prev_im), cv2.COLOR_RGB2GRAY)
                    cur_frm = cv2.cvtColor(np.array(cur_im), cv2.COLOR_RGB2GRAY)
                    of = cv2.calcOpticalFlowFarneback(prev_frm, cur_frm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    assert not np.isnan(np.sum(of)), 'nan found in flow, path: {}, idx: {}'.format(path, i)
                    assert not np.isinf(np.sum(of)), 'inf found in flow, path: {}, idx: {}'.format(path, i)
                    of = np.transpose(of, (2, 0, 1))
                    of = torch.from_numpy(of).float()
                    of = self.flow_transform(of)
                    flow_cub[i-1] = of
                prev_im = cur_im

            im_cub = im_cub.permute(1, 0, 2, 3)
            flow_cub = flow_cub.permute(1, 0, 2, 3)

            wrong_index = random.choice(
                [x for x in range(self.__len__()) if x != index])

            right_lmss = self.train_data[index]["lms_path"]
            right_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(right_lmss):
                right_embed[0,i * 4: 4 *(i +1 ),:] = torch.FloatTensor(np.load(lms_path))
            landmarks = example_lip

            wrong_lmss = self.train_data[wrong_index]["lms_path"]
            wrong_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(wrong_lmss):
                wrong_embed[0,i * 4 : (i +1) * 4,:] = torch.FloatTensor(np.load(lms_path))

            return example_lip, im_cub, landmarks, right_embed, wrong_embed, flow_cub

        else:
            paths = self.test_data[index]["image_path"]

            while True:
                a = random.randint(0,75)
                if paths[0][-8] == '_':
                    example_path = paths[0][:-8] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
                elif paths[0][-9] == '_':
                    example_path = paths[0][:-9] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
            example_lip = Image.open(example_path).convert("RGB").resize(self.output_shape)
            example_lip = self.transform(example_lip)

            im_cub = torch.FloatTensor(16, 3, self.output_shape[0], self.output_shape[1])
            flow_cub = torch.FloatTensor(15, 2, self.output_shape[0], self.output_shape[1])

            prev_im = None
            for i, path in enumerate(paths):
                im = Image.open(path).convert("RGB").resize(self.output_shape)
                cur_im = im
                im = self.transform(im)
                im_cub[i] = im

                # estimate optical flow
                if i != 0:
                    prev_frm = cv2.cvtColor(np.array(prev_im), cv2.COLOR_RGB2GRAY)
                    cur_frm = cv2.cvtColor(np.array(cur_im), cv2.COLOR_RGB2GRAY)
                    of = cv2.calcOpticalFlowFarneback(prev_frm, cur_frm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    of = np.transpose(of, (2, 0, 1))
                    of = torch.from_numpy(of).float()
                    of = self.flow_transform(of)
                    flow_cub[i-1] = of

                prev_im = cur_im

            im_cub = im_cub.permute(1, 0, 2, 3)
            flow_cub = flow_cub.permute(1, 0, 2, 3)

            right_lmss = self.test_data[index]["lms_path"]
            right_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(right_lmss):
                right_embed[0,i * 4: 4 *(i +1 ),:] = torch.FloatTensor(np.load(lms_path))
            landmarks = torch.FloatTensor(1, 16 , 20,2)
            for i,lms_path in enumerate(right_lmss):

                temp = np.load(lms_path.replace('lms','landmark'))[48:]
                original = np.sum(temp,axis=0) / 20.0
                temp = temp - original

                landmarks[0,i,:,:] = torch.FloatTensor(temp)

            caption = self.test_data[index]["lms_path"]


            return example_lip,im_cub, landmarks, right_embed, caption, flow_cub

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

class VaganFlowDataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[64, 64],
                 train=True):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = output_shape

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train:
            _file = open(os.path.join(dataset_dir, "train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        else:
            _file = open(os.path.join(dataset_dir, "test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.flow_transform = transforms.Normalize(mean=[0, 0], std=[20, 20])

    def __getitem__(self, index):
        if self.train:
            paths = self.train_data[index]["image_path"]
        else:
            paths = self.test_data[index]["image_path"]

        im_cub = torch.FloatTensor(16, 3, self.output_shape[0], self.output_shape[1])
        flow_cub = torch.FloatTensor(15, 2, self.output_shape[0], self.output_shape[1])

        prev_im = None
        for i, path in enumerate(paths):
            im = Image.open(path).convert("RGB").resize(self.output_shape)
            cur_im = im
            im = self.transform(im)
            im_cub[i] = im

            # estimate optical flow
            if i != 0:
                prev_frm = cv2.cvtColor(np.array(prev_im), cv2.COLOR_RGB2GRAY)
                cur_frm = cv2.cvtColor(np.array(cur_im), cv2.COLOR_RGB2GRAY)
                of = cv2.calcOpticalFlowFarneback(prev_frm, cur_frm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                of = np.transpose(of, (2, 0, 1))
                of = torch.from_numpy(of).float()
                of = self.flow_transform(of)
                flow_cub[i-1] = of

            prev_im = cur_im

        im_cub = im_cub.permute(1, 0, 2, 3)
        flow_cub = flow_cub.permute(1, 0, 2, 3)
        return im_cub, flow_cub

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class LRWdataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[64, 64],
                 train=True):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = output_shape

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train:
            _file = open(os.path.join(dataset_dir, "train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        else:
            _file = open(os.path.join(dataset_dir, "test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()

        self.transform = transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text
        if self.train:
            paths =  self.train_data[index]["image_path"]

            while True:
                a = random.randint(0,29)
                if paths[0][-8] == '_':
                    example_path = paths[0][:-8] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
                elif paths[0][-9] == '_':
                    example_path = paths[0][:-9] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
            example_lip = Image.open(example_path).convert("RGB").resize(self.output_shape)
            example_lip = self.transform(example_lip)
            # example_lip = example_lip.view(1,example_lip[0],example_lip[1],example_lip[2])

            im_cub = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
            for i,path in enumerate(paths):

                im = Image.open(path).convert("RGB").resize(self.output_shape)
                im = self.transform(im)
                im_cub[i,:,:,:] = im
            im_cub = im_cub.permute(1,0,2,3)
            wrong_index = random.choice(
                [x for x in range(self.__len__()) if x != index])

            right_lmss = self.train_data[index]["lms_path"]
            right_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(right_lmss):
                right_embed[0,i * 4: 4 *(i +1 ),:] = torch.FloatTensor(np.load(lms_path))

            # landmarks = torch.FloatTensor( 1, 16 , 20, 2)
            # for i,lms_path in enumerate(right_lmss):
            #     temp = np.load(lms_path.replace('lms','landmark'))[48:]
            #     original = np.sum(temp,axis=0) / 20.0
            #     temp = temp - original

            #     landmarks[0,i,:,:] = torch.FloatTensor(temp)

            wrong_lmss = self.train_data[wrong_index]["lms_path"]
            wrong_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(wrong_lmss):
                wrong_embed[0,i * 4 : (i +1) * 4,:] = torch.FloatTensor(np.load(lms_path))

            return example_lip,im_cub, example_lip, right_embed, wrong_embed
            # return example_lip,im_cub, right_embed, wrong_embed

        else:
            paths = self.test_data[index]["image_path"]

            while True:
                a = random.randint(0,29)
                if paths[0][-8] == '_':
                    example_path = paths[0][:-8] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
                elif paths[0][-9] == '_':
                    example_path = paths[0][:-9] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
            example_lip = Image.open(example_path).convert("RGB").resize(self.output_shape)
            example_lip = self.transform(example_lip)

            im_cub = torch.FloatTensor(16,3,self.output_shape[0],self.output_shape[1])
            for i,path in enumerate(paths):
                im = Image.open(path).convert("RGB").resize(self.output_shape)
                im = self.transform(im)
                im_cub[i,:,:,:] = im
            im_cub = im_cub.permute(1,0,2,3)
            right_lmss = self.test_data[index]["lms_path"]
            right_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(right_lmss):
                right_embed[0,i * 4: 4 *(i +1 ),:] = torch.FloatTensor(np.load(lms_path))
            # landmarks = torch.FloatTensor(1, 16 , 20,2)
            # for i,lms_path in enumerate(right_lmss):

            #     temp = np.load(lms_path.replace('lms','landmark'))[48:]
            #     original = np.sum(temp,axis=0) / 20.0
            #     temp = temp - original

            #     landmarks[0,i,:,:] = torch.FloatTensor(temp)

            caption = self.test_data[index]["lms_path"]


            return example_lip,im_cub, example_lip, right_embed, caption
            # return example_lip,im_cub, right_embed, caption

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)


class LRWFarnebackDataset(data.Dataset):
    def __init__(self,
                 dataset_dir,
                 output_shape=[64, 64],
                 train=True):
        self.train = train
        self.dataset_dir = dataset_dir
        self.output_shape = output_shape

        if not len(output_shape) in [2, 3]:
            raise ValueError("[*] output_shape must be [H,W] or [C,H,W]")

        if self.train:
            _file = open(os.path.join(dataset_dir, "train.pkl"), "rb")
            self.train_data = pickle.load(_file)
            _file.close()
        else:
            _file = open(os.path.join(dataset_dir, "test.pkl"), "rb")
            self.test_data = pickle.load(_file)
            _file.close()

        self.transform = transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.flow_transform = transforms.Normalize(mean=[0, 0], std=[20, 20])

    def __getitem__(self, index):
        # In training phase, it return real_image, wrong_image, text

        if self.train:
            paths =  self.train_data[index]["image_path"]

            while True:
                a = random.randint(0,29)
                if paths[0][-8] == '_':
                    example_path = paths[0][:-8] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
                elif paths[0][-9] == '_':
                    example_path = paths[0][:-9] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
            example_lip = Image.open(example_path).convert("RGB").resize(self.output_shape)
            example_lip = self.transform(example_lip)
            # example_lip = example_lip.view(1,example_lip[0],example_lip[1],example_lip[2])

            im_cub = torch.FloatTensor(16, 3, self.output_shape[0], self.output_shape[1])
            flow_cub = torch.FloatTensor(15, 2, self.output_shape[0], self.output_shape[1])

            prev_im = None
            for i, path in enumerate(paths):
                im = Image.open(path).convert("RGB").resize(self.output_shape)
                cur_im = im
                im = self.transform(im)
                im_cub[i] = im

                # estimate optical flow
                if i != 0:
                    prev_frm = cv2.cvtColor(np.array(prev_im), cv2.COLOR_RGB2GRAY)
                    cur_frm = cv2.cvtColor(np.array(cur_im), cv2.COLOR_RGB2GRAY)
                    of = cv2.calcOpticalFlowFarneback(prev_frm, cur_frm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    assert not np.isnan(np.sum(of)), 'nan found in flow, path: {}, idx: {}'.format(path, i)
                    assert not np.isinf(np.sum(of)), 'inf found in flow, path: {}, idx: {}'.format(path, i)
                    of = np.transpose(of, (2, 0, 1))
                    of = torch.from_numpy(of).float()
                    of = self.flow_transform(of)
                    flow_cub[i-1] = of
                prev_im = cur_im

            im_cub = im_cub.permute(1, 0, 2, 3)
            flow_cub = flow_cub.permute(1, 0, 2, 3)

            wrong_index = random.choice(
                [x for x in range(self.__len__()) if x != index])

            right_lmss = self.train_data[index]["lms_path"]
            right_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(right_lmss):
                right_embed[0,i * 4: 4 *(i +1 ),:] = torch.FloatTensor(np.load(lms_path))

            # landmarks = torch.FloatTensor( 1, 16 , 20, 2)
            # for i,lms_path in enumerate(right_lmss):
            #     temp = np.load(lms_path.replace('lms','landmark'))[48:]
            #     original = np.sum(temp,axis=0) / 20.0
            #     temp = temp - original

            #     landmarks[0,i,:,:] = torch.FloatTensor(temp)

            wrong_lmss = self.train_data[wrong_index]["lms_path"]
            wrong_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(wrong_lmss):
                wrong_embed[0,i * 4 : (i +1) * 4,:] = torch.FloatTensor(np.load(lms_path))

            return example_lip,im_cub, example_lip, right_embed, wrong_embed, flow_cub
            # return example_lip,im_cub, right_embed, wrong_embed

        else:
            paths = self.test_data[index]["image_path"]

            while True:
                a = random.randint(0,29)
                if paths[0][-8] == '_':
                    example_path = paths[0][:-8] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
                elif paths[0][-9] == '_':
                    example_path = paths[0][:-9] +'_%03d.jpg' %(a)
                    if os.path.isfile(example_path):
                        break
            example_lip = Image.open(example_path).convert("RGB").resize(self.output_shape)
            example_lip = self.transform(example_lip)

            im_cub = torch.FloatTensor(16, 3, self.output_shape[0], self.output_shape[1])
            flow_cub = torch.FloatTensor(15, 2, self.output_shape[0], self.output_shape[1])

            prev_im = None
            for i, path in enumerate(paths):
                im = Image.open(path).convert("RGB").resize(self.output_shape)
                cur_im = im
                im = self.transform(im)
                im_cub[i] = im

                # estimate optical flow
                if i != 0:
                    prev_frm = cv2.cvtColor(np.array(prev_im), cv2.COLOR_RGB2GRAY)
                    cur_frm = cv2.cvtColor(np.array(cur_im), cv2.COLOR_RGB2GRAY)
                    of = cv2.calcOpticalFlowFarneback(prev_frm, cur_frm, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    of = np.transpose(of, (2, 0, 1))
                    of = torch.from_numpy(of).float()
                    of = self.flow_transform(of)
                    flow_cub[i-1] = of

                prev_im = cur_im

            im_cub = im_cub.permute(1, 0, 2, 3)
            flow_cub = flow_cub.permute(1, 0, 2, 3)

            right_lmss = self.test_data[index]["lms_path"]
            right_embed = torch.FloatTensor(1,64,128)
            for i,lms_path in enumerate(right_lmss):
                right_embed[0,i * 4: 4 *(i +1 ),:] = torch.FloatTensor(np.load(lms_path))
            # landmarks = torch.FloatTensor(1, 16 , 20,2)
            # for i,lms_path in enumerate(right_lmss):

            #     temp = np.load(lms_path.replace('lms','landmark'))[48:]
            #     original = np.sum(temp,axis=0) / 20.0
            #     temp = temp - original

            #     landmarks[0,i,:,:] = torch.FloatTensor(temp)

            caption = self.test_data[index]["lms_path"]


            return example_lip,im_cub, example_lip, right_embed, caption, flow_cub
            # return example_lip,im_cub, right_embed, caption

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
