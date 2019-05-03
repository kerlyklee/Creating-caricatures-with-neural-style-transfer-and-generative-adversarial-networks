#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import os
import random
from keras.preprocessing.image import img_to_array
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
import numpy as np
from imutils import paths

TRAIN_FILE = "train.p"
VALID_FILE = "valid.p"
TEST_FILE = "test.p"


def mk_square(img, desired_shp=1000):
    x, y, _ = img.shape
    maxs = max(img.shape[:2])
    y2 = (maxs-y)//2
    x2 = (maxs-x)//2
    arr = np.zeros((maxs, maxs, 3), dtype=np.float32)
    arr[int(np.floor(x2)):int(np.floor(x2)+x), int(np.floor(y2)):int(np.floor(y2)+y)] = img
    return imresize(arr, (desired_shp, desired_shp))


def get_data_images(input):
    images = []
    labels = []

    image_paths = sorted(list(paths.list_images(input)))
    random.seed(42)
    random.shuffle(images)

    i = 0

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = img_to_array(image)
        i = i + 1
        print('[INFO] reading in image %s...' % (str(i)))
        sq_big_image = mk_square(image)
        label = image_path.split(os.path.sep)[-2]
        sq_l_image = cv2.resize(sq_big_image, (32, 32))
        label = 1 if label == 'portrait' else 0
        labels.append(label)
        images.append(sq_l_image)

    images = np.array(images, dtype='float') / 255.0
    labels = np.array(labels)

    (X_train, X_test, Y_train, Y_test) = train_test_split(images, labels, test_size=0.3, random_state=42)

    (X_test, X_valid, Y_test, Y_valid) = train_test_split(X_test, Y_test, test_size=0.33, random_state=42)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test
