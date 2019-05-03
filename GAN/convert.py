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
extension = ".png"
for file in os.listdir("C:/Users/Admin/Thesis/GAN/Caricature_Data"):
    if file.endswith(".jpg"):
    	#to remove jpg files after converting:
    	os.remove(file)
    	''' to change jpg to png 
        im = Image.open(file)
        im.save(file + extension)
		'''