import keras.backend as K
from keras.applications import VGG16

from PIL import Image

import numpy as np
import time

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

import argparse
import sys; sys.argv=['']; del sys


parser = argparse.ArgumentParser(description='Image neural style transfer implemented with Keras')
#parser.add_argument('C:/Users/Kerly/Thesis/NeuralStyleTransfer/KatsePildid/city', metavar='content', type=str, help='Path to target content image')
#parser.add_argument('C:/Users/Kerly/Thesis/NeuralStyleTransfer/KatsePildid/style1', metavar='style', type=str, help='Path to target style image')
#parser.add_argument('Tulemus1', metavar='res_prefix', type=str, help='Name of generated image')
parser.add_argument('--iter', type=int, default=10, required=False, help='Number of iterations to run')
parser.add_argument('--content_weight', type=float, default=0.025, required=False, help='Content weight')
parser.add_argument('--style_weight', type=float, default=1.0, required=False, help='Style weight')
parser.add_argument('--var_weight', type=float, default=1.0, required=False, help='Total Variation weight')
parser.add_argument('--height', type=int, default=512, required=False, help='Height of the images')
parser.add_argument('--width', type=int, default=512, required=False, help='Width of the images')

args = parser.parse_args()
