#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Test the model
"""
import argparse
import cv2
import numpy as np
import os
from keras.preprocessing.image import img_to_array
from scipy.misc import imresize
from model import ModelHumorPrediction


def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=True, help='Path to dataset')
    parser.add_argument('-m', '--model', required=True, help='Path to output model')
    parser.add_argument('-s', '--size', type=int, default=32, help='Width x height for resizing image (square), '
                                                                   'should be the same shape as in trained model')
    parser_args = vars(parser.parse_args())
    return parser_args


def mk_square(img, desired_shp=1000):
    x, y, _ = img.shape
    maxs = max(img.shape[:2])
    y2 = (maxs-y)//2
    x2 = (maxs-x)//2
    arr = np.zeros((maxs, maxs, 3), dtype=np.float32)
    arr[int(np.floor(x2)):int(np.floor(x2)+x), int(np.floor(y2)):int(np.floor(y2)+y)] = img
    return imresize(arr, (desired_shp , desired_shp))


def load_image(image_path, args):
    images = []

    image = cv2.imread(image_path)
    image = img_to_array(image)
    sq_big_image = mk_square(image)
    sq_l_image = cv2.resize(sq_big_image, (args['size'], args['size']))
    images.append(sq_l_image)
    images = np.array(images, dtype='float') / 255.0
    return images


def load_original_image(args):
    image = cv2.imread(args['image'])
    return image.copy()


def write_to_file(name, label):
    f = open("labels_50_capsnet.txt", "a+")
    class_name = name[0]
    fls = ''
    if class_name == 'c':
        if 'Karikatuur' not in label:
            fls = 'VALESTI C'
    if class_name == 'p':
        if 'Portree' not in label:
            fls = 'VALESTI P'
    f.write("%s Label: %s, name: %s \n" % (fls, label, name))
    f.close()


def test_images(arguments):
    path = arguments['dataset']
    model = ModelHumorPrediction("Humor Prediction", output_folder=None)
    model.load(arguments['model'])
    for image_path in os.listdir(path):
        img_path = os.path.join(path, image_path)
        images = load_image(img_path, arguments)
        predictions, classes = model.predict(images)
        caricature = predictions[0][0]
        portrait = predictions[0][1]
        label = 'Karikatuur' if caricature > portrait else 'Portree'
        write_to_file(image_path, label)


if __name__ == '__main__':
    arguments = load_arguments()
    test_images(arguments)
