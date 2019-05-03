# resize GAN.py
# code inspired from https://github.com/llSourcell/Pokemon_GAN
import os
import cv2

src = "./Caricature_Data" #pokeRGB_black
dst = "./sized_data" # resized

os.mkdir(dst)

for each in os.listdir(src):
    img = cv2.imread(os.path.join(src,each))
    img = cv2.resize(img,(256,256))
    cv2.imwrite(os.path.join(dst,each), img)
    