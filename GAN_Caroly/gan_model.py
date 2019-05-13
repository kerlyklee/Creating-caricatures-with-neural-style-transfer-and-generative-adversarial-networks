
# generate caricatures
# code inspierd from https://github.com/llSourcell/Pokemon_GAN
import os
import tensorflow as tf
import numpy as np
import cv2
import random
import scipy.misc
from utils import *
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import argparse
import utils
from scipy.misc import imresize
import matplotlib.pyplot as plt
import sys; sys.argv=['']; del sys
from tensorflow.contrib import predictor
import torch
import PIL
from PIL import Image
from io import StringIO
from pandas.compat import StringIO



HEIGHT, WIDTH, CHANNEL = 128, 128, 3
BATCH_SIZE = 64
EPOCH = 251
version = 'new_caricatures9'
newCaric_path = './' + version


# lrelu function
def lrelu(x, n, leak=0.2): 
    return tf.maximum(x, leak * x, name=n) 
# get rigth caricature data
'''def process_data():   
    current_dir = os.getcwd()
    caricature_dir = os.path.join(current_dir, 'Caricature_Data')
    images = []
    for each in os.listdir(caricature_dir): 
        images.append(os.path.join(caricature_dir,each))
    # print images    
    all_images = tf.convert_to_tensor(images, dtype = tf.string)
    
    images_queue = tf.train.slice_input_producer([all_images])
                                        
    content = tf.read_file(images_queue[0])
    image = tf.image.decode_image(content, channels = CHANNEL)
    #change to get more difrent
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta = 0.1)
    image = tf.image.random_contrast(image, lower = 0.9, upper = 1.1)

    size = [HEIGHT, WIDTH]
    image = tf.image.resize_images(image, size)
    image.set_shape([HEIGHT,WIDTH,CHANNEL])

    #change tensor type
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    
    iamges_batch = tf.train.shuffle_batch(
                                    [image], batch_size = BATCH_SIZE,
                                    num_threads = 4, capacity = 200 + 3* BATCH_SIZE,
                                    min_after_dequeue = 200)
    num_images = len(images)

    return iamges_batch, num_images '''

def generator(input, random_dim, reuse=False):
    c4, c8, c16, c32, c64 = 512, 256, 128, 64, 36 # channel num
    s4 = 4
    output_dim = CHANNEL  # RGB image
    with tf.variable_scope('gens') as scope:
        if reuse:
            scope.reuse_variables()
        w1 = tf.get_variable('w1', shape=[random_dim, s4 * s4 * c4], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        print (w1)
        b1 = tf.get_variable('b1', shape=[c4 * s4 * s4], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))
        print(b1)
        flat_conv1 = tf.add(tf.matmul(input, w1), b1, name='flat_conv1')
         #Convolution, bias, activation, repeat! 
        conv1 = tf.reshape(flat_conv1, shape=[-1, s4, s4, c4], name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training=True, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn1')
        act1 = tf.nn.relu(bn1, name='act1')
        # 8*8*256
        #Convolution, bias, activation, repeat! 
        conv2 = tf.layers.conv2d_transpose(act1, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=True, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = tf.nn.relu(bn2, name='act2')
        # 16*16*128
        conv3 = tf.layers.conv2d_transpose(act2, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=True, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = tf.nn.relu(bn3, name='act3')
        # 32*32*64
        conv4 = tf.layers.conv2d_transpose(act3, c32, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=True, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = tf.nn.relu(bn4, name='act4')
        # 64*64*32
        conv5 = tf.layers.conv2d_transpose(act4, c64, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv5')
        bn5 = tf.contrib.layers.batch_norm(conv5, is_training=True, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn5')
        act5 = tf.nn.relu(bn5, name='act5')
        
        #128*128*3
        conv6 = tf.layers.conv2d_transpose(act5, output_dim, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                           kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                           name='conv6')
        # bn6 = tf.contrib.layers.batch_norm(conv6, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn6')
        act6 = tf.nn.tanh(conv6, name='act6')
        return act6



#def discriminator(input, is_train, reuse=False):


        #output = imutils.resize(orig, width=400)
        #cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        #cv2.imshow('Image with predicted label', output)
        #cv2.imwrite('../dataset/1.jpg', output)
        #cv2.waitKey(0)
    '''c2, c4, c8, c16 = 64, 128, 256, 512  # channel num: 64, 128, 256, 512
    with tf.variable_scope('dis') as scope:
        if reuse:
            scope.reuse_variables()

        #Convolution, activation, bias, repeat! 
        conv1 = tf.layers.conv2d(input, c2, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv1')
        bn1 = tf.contrib.layers.batch_norm(conv1, is_training = is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope = 'bn1')
        act1 = lrelu(conv1, n='act1')
         #Convolution, activation, bias, repeat! 
        conv2 = tf.layers.conv2d(act1, c4, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv2')
        bn2 = tf.contrib.layers.batch_norm(conv2, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn2')
        act2 = lrelu(bn2, n='act2')
        #Convolution, activation, bias, repeat! 
        conv3 = tf.layers.conv2d(act2, c8, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv3')
        bn3 = tf.contrib.layers.batch_norm(conv3, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn3')
        act3 = lrelu(bn3, n='act3')
         #Convolution, activation, bias, repeat! 
        conv4 = tf.layers.conv2d(act3, c16, kernel_size=[5, 5], strides=[2, 2], padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                                 name='conv4')
        bn4 = tf.contrib.layers.batch_norm(conv4, is_training=is_train, epsilon=1e-5, decay = 0.9,  updates_collections=None, scope='bn4')
        act4 = lrelu(bn4, n='act4')
       
        # start from act4
        dim = int(np.prod(act4.get_shape()[1:]))
        fc1 = tf.reshape(act4, shape=[-1, dim], name='fc1')
      
        
        w2 = tf.get_variable('w2', shape=[fc1.shape[-1], 1], dtype=tf.float32,
                             initializer=tf.truncated_normal_initializer(stddev=0.02))
        b2 = tf.get_variable('b2', shape=[1], dtype=tf.float32,
                             initializer=tf.constant_initializer(0.0))

 
        logits = tf.add(tf.matmul(fc1, w2), b2, name='logits')
        # dcgan
        acted_out = tf.nn.sigmoid(logits)'''
        #return logits #, acted_out


def train():
    #G(z)
    random_dim = 3
    loss_const = 3
    def load_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument('-m', '--model', default='C:/Users/Admin/Thesis/GAN_Caroly/saved_model.pbtxt', type=str, help='Path to model')

        args = vars(parser.parse_args())


        return args


    def mk_square(img, desired_shp=1000):
        x, y, _ = img.shape
        maxs = max(img.shape[:2])
        y2 = (maxs-y)//2
        x2 = (maxs-x)//2
        arr = np.zeros((maxs, maxs, 3), dtype=np.float32)
        arr[int(np.floor(x2)):int(np.floor(x2)+x), int(np.floor(y2)):int(np.floor(y2)+y)] = img
        return imresize(arr, (desired_shp , desired_shp))


    def load_image(gen_img):
        image = img_to_array(gen_img)
        image = mk_square(image)
        image = cv2.resize(image, (28, 28))
        image = image.astype('float32') / 255.0
        #plt.imshow(image)
        image = np.expand_dims(image, axis=0)
        return image



    '''def load_original_image(args):
        image = cv2.imread(args['image'])
        return image.copy()'''


    def load_trained_model(args):
        print('[INFO] loading model...')
        model = load_model(args['model'])
        return model


    def classify_image(model, image):
        (caricature, nonsenss) = model.predict(image)[0]
        probability = caricature if caricature > nonsenss else nonsenss
        #print('Probability: ' + str(probability))
        #label = '{}: {:.2f}%'.format(label, probability * 100)
        return probability

    def tensor_into_image(fake_image):
        type_int = tf.cast(fake_image, dtype=tf.int32)
        image_data = type_int[0]
        print (image_data)

        gen_image = tf.cast(image_data, dtype=tf.uint8)
        #your_session = tf.Session()
        #a_np=image_data.eval(session=your_session)
        sess = tf.Session()
        #saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        #init = tf.global_variables_initializer()
        

        #tf.InteractiveSession()
        a_np=image_data.eval(session=sess)
       
        return a_np
        #your_session = tf.Session()
        #array = tensor.eval(session=your_session)

        #jpeg_bin_tensor = tf.image.encode_jpeg(gen_image)


    
    with tf.variable_scope('input'):
        #real and fake image placholders
        #real_image = tf.placeholder(tf.float32, shape = [None, HEIGHT, WIDTH, CHANNEL], name='real_image')
        #random_input = tf.placeholder(tf.float32, shape=[None, random_dim], name='rand_input')
        #is_train = tf.placeholder(tf.bool, name='is_train')
        path = "algus1.png"
        image = cv2.imread(path)
        image = image[0]
        start_image = tf.convert_to_tensor(image, dtype=tf.float32)

    fake_image = generator(start_image, random_dim)
    needed_data = tensor_into_image(fake_image)

    '''predict_fn = predictor.from_saved_model('C:/Users/Admin/Thesis/GAN_Caroly')
    predict_fn = a.encode('utf-8').strip()
    predictions = predict_fn(fake_image)
    print(predictions)'''
    fake_img = load_image(needed_data)
    args = load_arguments()
    model = load_trained_model(args)
    loss = classify_image(model, fake_img)
    g_loss = tf.reduce_mean((1 - loss)*fake_image)
    print (g_loss)
   


    #real_result = discriminator(real_image, is_train)

    #fake_result = discriminator(fake_image, is_train, reuse=tf.AUTO_REUSE)
    
    #d_loss = tf.reduce_mean(fake_result) - tf.reduce_mean(loss*loss_const)  # This optimizes the dektective
    #g_loss = -tf.reduce_mean(d_loss) 
    #g_loss = tf.cast(g_loss, dtype=tf.float32)
    #print (g_loss)  
    t_vars = tf.trainable_variables()
    #d_vars = [var for var in t_vars if 'dis' in var.name]
    g_vars = [var for var in t_vars if 'gens' in var.name]
    #trainer_d = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(d_loss, var_list=d_vars)
    #trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    trainer_g = tf.train.RMSPropOptimizer(learning_rate=2e-4).minimize(g_loss, var_list=g_vars)
    # clip discriminator weights
    #d_clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in d_vars]
    batch_size = BATCH_SIZE
    #image_batch, samples_num = process_data()  

    batch_num = int(400 / batch_size)
    total_batch = 0

            


    sess = tf.Session()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # continue training
    save_path = saver.save(sess, "/tmp/model.ckpt")
    ckpt = tf.train.latest_checkpoint('./model/' + version)
    saver.restore(sess, save_path)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    #print('total training sample num:%d' % samples_num)
    #print('batch size: %d, batch num per epoch: %d, epoch num: %d' % (batch_size, batch_num, EPOCH))
    print('start training...')
    for i in range(EPOCH):
        print("Running epoch {}/{}...".format(i, EPOCH))
        for j in range(batch_num):
            print(j)
            #d_iters = 5
            g_iters = 1

            train_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            '''for k in range(d_iters):
                print(k)
                train_image = sess.run(image_batch)
                #clip weights
                sess.run(d_clip)
                
                # Update the discriminator
                _, dLoss = sess.run([trainer_d, d_loss],
                                    feed_dict={random_input: train_noise, real_image: train_image, is_train: True})'''

            # Update the generator
            for k in range(g_iters):
                _, gLoss = sess.run([trainer_g, g_loss],
                                   feed_dict={start_image: train_noise})
            
        # save check point for model every 500 epoch
        if i%500 == 0:
            if not os.path.exists('./model/' + version):
                os.makedirs('./model/' + version)
            saver.save(sess, './model/' +version + '/' + str(i))  
        if i%50 == 0:
            # save image after every 50 epochs
            if not os.path.exists(newCaric_path):
                os.makedirs(newCaric_path)
            sample_noise = np.random.uniform(-1.0, 1.0, size=[batch_size, random_dim]).astype(np.float32)
            imgtest = sess.run(fake_image, feed_dict={start_image: sample_noise})
            save_images(imgtest, [8,8] ,newCaric_path + '/epoch' + str(i) + '.jpg')
            
            print('train:[%d],g_loss:%f' % (i, gLoss))
    coord.request_stop()
    coord.join(threads)


if __name__ == "__main__":

    train()
    # test()

