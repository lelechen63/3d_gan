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
def crop_lips(image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('/home/lele/Desktop/hehe/shape_predictor_68_face_landmarks.dat')
    image = cv2.imread(image_path)
    print image.shape
    # image = cv2.resize(image, (558, 992), interpolation = cv2.INTER_AREA)
    # cv2.imwrite(image_path, image)

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
            print roi.shape
            cv2.imshow("cropped", roi)
            cv2.waitKey(0)
            cv2.imwrite('/home/lele/Desktop/hehe/trump3_roi.jpg', roi)
            return shape,roi





image_path='/home/lele/Desktop/hehe/trump3.jpg'
crop_lips(image_path)