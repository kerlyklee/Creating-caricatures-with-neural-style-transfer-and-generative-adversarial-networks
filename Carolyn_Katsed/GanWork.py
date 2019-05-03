import numpy as np 
import matplotlib.pyplot as plt
from generator import Generator 
from test_one_picture import Discriminator
import argparse
from ops import *
import os
import sys; sys.argv=['']; del sys
import keras.backend as K
from keras.applications import VGG16
from PIL import Image
import time
from imutils import paths
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import argparse
import sys; sys.argv=['']; del sys
import tensorflow as tf

class GAN:
    def __init__(self, img_shape, epochs=50000, lr_gen=0.0001, lr_disc=0.0001, z_shape=100, batch_size=64, beta1=0.5, epochs_for_sample=500):
        
       
        self.rows, self.cols, self.channels = img_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.z_shape = z_shape
        self.epochs_for_sample = epochs_for_sample
        self.generator = Generator(img_shape, self.batch_size)
        self.discriminator = Discriminator(img_shape)
        parser = argparse.ArgumentParser()
        parser.add_argument('--style_img', default='C:/Users/Admin/Thesis/NeuralStyleTransfer/KatsePildid/Barac_Obama.jpg', type=str, help='Path to target style image')
        parser.add_argument('--dataset', default='C:/Users/Admin/Thesis/GAN/dataset_50', type=str, help='Path to  dataset')
        parser.add_argument('--height', type=int, default=1051, required=False, help='Height of the images')
        parser.add_argument('--width', type=int, default=700, required=False, help='Width of the images')
        args = parser.parse_args()
        img_height = args.height
        img_width = args.width
        style_path = args.style_img
       
        image_paths = sorted(list(paths.list_images(args['dataset'])))
      
        def process_img(path):
            """
           Formatting the image
            """
            # Open image and resize
            img = Image.open(path)
            img = img.resize((img_width, img_height))

            # Convert  to data array
            data = np.asarray(img, dtype='float32')
            data = np.expand_dims(data, axis=0)
            data = data[:, :, :, :3]

            # Apply pre-process to match VGG16 we are using
            data[:, :, :, 0] -= 103.939
            data[:, :, :, 1] -= 116.779
            data[:, :, :, 2] -= 123.68

            # Flip from RGB to BGR
            data = data[:, :, :, ::-1]

            return data 
        style = process_img(style_path)
        images = process_img(image_paths)       # img_shape = args.picture

        (x_train, _) = images
        (x_test, _) = style

        X = np.concatenate([x_train, x_test])
        self.X = X / 127.5 - 1 # Scale between -1 and 1
        self.phX = tf.placeholder(tf.float32, [None, self.rows, self.cols])
        self.phZ = tf.placeholder(tf.float32, [None, self.z_shape])
    
        self.gen_out = self.generator.forward(self.phZ)

        disc_logits_fake = self.discriminator.forward(self.gen_out)
        disc_logits_real = self.discriminator.forward(self.phX)

        disc_fake_loss = cost(tf.zeros_like(disc_logits_fake), disc_logits_fake)
        disc_real_loss = cost(tf.ones_like(disc_logits_real), disc_logits_real)

        self.disc_loss = tf.add(disc_fake_loss, disc_real_loss)
        self.gen_loss = cost(tf.ones_like(disc_logits_fake), disc_logits_fake)

        train_vars = tf.trainable_variables()

        disc_vars = [var for var in train_vars if 'd' in var.name]
        gen_vars = [var for var in train_vars if 'g' in var.name]

        self.disc_train = tf.train.AdamOptimizer(lr_disc,beta1=beta1).minimize(self.disc_loss, var_list=disc_vars)
        self.gen_train = tf.train.AdamOptimizer(lr_gen, beta1=beta1).minimize(self.gen_loss, var_list=gen_vars)
        


    def train(self):
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        for i in range(self.epochs):
            idx = np.random.randint(0, len(self.X), self.batch_size)
            batch_X = self.X[idx]
            batch_Z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))


            _, d_loss = self.sess.run([self.disc_train, self.disc_loss], feed_dict={self.phX:batch_X, self.phZ:batch_Z})
            batch_Z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
            _, g_loss = self.sess.run([self.gen_train, self.gen_loss], feed_dict={self.phZ: batch_Z})
            if i % self.epochs_for_sample == 0:
                self.generate_sample(i)
                print(f"Epoch: {i}. Discriminator loss: {d_loss}. Generator loss: {g_loss}")


    def generate_sample(self, epoch):
        c = 7
        r = 7
        z = np.random.uniform(-1, 1, (self.batch_size, self.z_shape))
        imgs = self.sess.run(self.gen_out, feed_dict={self.phZ:z})
        imgs = imgs*0.5 + 0.5
        # scale between 0, 1
        fig, axs = plt.subplots(c, r)
        cnt = 0
        for i in range(c):
            for j in range(r):
                axs[i, j].imshow(imgs[cnt, :, :, 0], cmap="gray")
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("samples/%d.png" % epoch)
        plt.close()





if __name__ == '__main__':
    img_shape = (28, 28, 1)
    epochs = 50000
    gan = GAN(img_shape, epochs)

    if not os.path.exists('samples/'):
        os.makedirs('samples/')
    
    gan.train()
    
   