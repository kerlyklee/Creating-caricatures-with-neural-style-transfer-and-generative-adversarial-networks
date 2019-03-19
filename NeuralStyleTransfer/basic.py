import keras.backend as K
from keras.applications import VGG16

from PIL import Image

import numpy as np 
import time

from scipy.optimize import fmin_1_bfgs_b
from scipy.misc import imsave

import argparse