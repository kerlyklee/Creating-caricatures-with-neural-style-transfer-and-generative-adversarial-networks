#!/usr/bin/env python3
from PIL import Image
from os import listdir
from os.path import splitext
from PIL import Image
import numpy as np
import sys
import os
import csv
import os
extension = ".jpg"
for file in os.listdir("C:/Users/Admin/Thesis/GAN_Caroly/Data/caricature"):
    if file.endswith(".png"):
    	name=file.replace('.png', '')
        im = Image.open(name)

        im.save(file + extension)
		