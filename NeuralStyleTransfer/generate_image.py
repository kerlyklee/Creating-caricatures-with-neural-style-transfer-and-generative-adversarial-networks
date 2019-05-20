# code based from https://github.com/PrzemekPobrotyn/Neural-Style-Transfer
from nst_utils import *
from neural_network import *
import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description=
        'Generates a new image with content similar to provided content image' +
        ' and in style of provided style image' +
        ' using neural style transfer')

    parser.add_argument("content_image_path", help='path to content image')
    parser.add_argument("style_image_path", help='path to style image')
    parser.add_argument("output_path", help='path for the ouput file')
    parser.add_argument('num_iterations', help ='number of iterations')
    parser.add_argument("-s", "--save", help='flag indicating whether to save intermediate values', action='store_true')
    return parser.parse_args()


args = parse_args()
content_image, style_image = resize_style_image(args.content_image_path, args.style_image_path)

CONFIG.IMAGE_WIDTH = content_image.shape[1]
CONFIG.IMAGE_HEIGHT = content_image.shape[0]

content_image = reshape_and_normalize_image(content_image) # actually, this isn't necessary
style_image = reshape_and_normalize_image(style_image)
sess, train_step, model = prepare_network(content_image, style_image)
train_network(sess, train_step, model, content_image, args.output_path,
              int(args.num_iterations), args.save)
