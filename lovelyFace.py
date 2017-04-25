import tensorflow as tf
import cv2
import glob
import os
import posixpath
import random
import numpy as np


def read_train_data_set():
    """
    
    :return: 
    """
    def get_image_score(filename):
        a = filename.split('\\')
        return int(a[-2])

    score_list = []
    img_list = []

    filelist = glob.glob('./evaluateurface/train/*/*.jpeg')
    random.shuffle(filelist)

    for file in filelist:
        file = os.path.abspath(file)
        img = cv2.imread(file)
        height, width, channels = img.shape
        if height == 200 and width == 200:
            score = get_image_score(file)
            score_list.append(score)
            img_list.append(img)
        else:
            print(file)

    return np.array(score_list), np.array(img_list)

score_array, b = read_train_data_set()
img_array = np.stack(b)

print(score_array.shape, img_array.shape)

print(score_array)

score_onehot = np.equal.outer(score_array, np.arange(1,6)).astype(np.float)
print(score_onehot.shape)