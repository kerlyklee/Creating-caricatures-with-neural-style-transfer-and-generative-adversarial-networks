#!/usr/bin/env python3
# code written by example https://github.com/hunter-heidenreich/ML-Open-Source-Implementations/blob/master/Style-Transfer/Style%20Transfer.ipynb
import keras.backend as K
from keras.applications import VGG16
from PIL import Image
import numpy as np
import time
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import argparse
import sys; sys.argv=['']; del sys
import cv2
import os
#C:/Users/Admin/Thesis/NeuralStyleTransfer/Caricature/kerlyklee@tensorflow-1-vm
parser = argparse.ArgumentParser(description='Creating Caricatures')
parser.add_argument('--cont_img', default='Thesis/NeuralStyleTransfer/Portrait/Beyonce.jpg', type=str, help='Path to original image')
parser.add_argument('--style_img', default='Thesis/NeuralStyleTransfer/Caricature/BeyonceC.jpg', type=str, help='Path to wished style image')
parser.add_argument('--result_img', default='tulemus', type=str, help='Name of generated images')
parser.add_argument('--iterations', type=int, default=250, required=False, help='Number of iterations to run')
parser.add_argument('--cont_weight', type=float, default=0.025, required=False, help='Content weight')
parser.add_argument('--style_weight', type=float, default=1.0, required=False, help='Style weight')
parser.add_argument('--var_weight', type=float, default=1.0, required=False, help='Total Variation weight')
parser.add_argument('--height', type=int, default=512, required=False, help='Height of the image')
parser.add_argument('--width', type=int, default=512, required=False, help='Width of the image')

args = parser.parse_args()

#used parameters
img_height = args.height
img_width = args.width
img_size = img_height * img_width
img_channels = 3

cont_path = args.cont_img
style_path = args.style_img
result_path = args.result_img
result_ext = '.jpg'

CONT_IMG_POS = 0
STYLE_IMG_POS = 1
RESULT_IMG_POS = 2

#greating usable data
def process_imgage(path):
    # Open image and resize
    img = Image.open(path)
    img = img.resize((img_width, img_height))

    # transform to data array
    data = np.asarray(img, dtype='float32')
    data = np.expand_dims(data, axis=0)
    data = data[:, :, :, :3]

    # matching VGG16
    data[:, :, :, 0] -= 103.939
    data[:, :, :, 1] -= 116.779
    data[:, :, :, 2] -= 123.68

    # Flip from RGB to BGR
    data = data[:, :, :, ::-1]

    return data

#getting needed Layers
def get_layers(cont_matrix, style_matrix, generated_matrix):
   
    # Prepear to new input sizes
    input_tensor = K.concatenate([cont_matrix, style_matrix, generated_matrix], axis=0)
    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

    # Change layers to dictionary
    layers = dict([(layer.name, layer.output) for layer in model.layers])

    # Take the specific layers
    c_layers = layers['block2_conv2']
    s_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    s_layers = [layers[layer] for layer in s_layers]

    return c_layers, s_layers

#for calculating loss 
def cont_loss(cont_features, generated_features):
 
    return 0.5 * K.sum(K.square(generated_features - cont_features))

#for calculating gram matrix for style loss
def gram_matrix(features):

    return K.dot(features, K.transpose(features))

#for calculating style loss
def style_loss(style_matrix, generated_matrix):

    style_features = K.batch_flatten(K.permute_dimensions(style_matrix, (2, 0, 1)))
    generated_features = K.batch_flatten(K.permute_dimensions(generated_matrix, (2, 0, 1)))

    # Gets gram matrix
    style_mat = gram_matrix(style_features)
    generated_mat = gram_matrix(generated_features)

    return K.sum(K.square(style_mat - generated_mat)) / (4.0 * (img_channels ** 2) * (img_size ** 2))

#for calculating variation loss 
def variation_loss(generated_matrix):

    a = K.square(generated_matrix[:, :img_height-1, :img_width-1, :] - generated_matrix[:, 1:, :img_width-1, :])
    b = K.square(generated_matrix[:, :img_height-1, :img_width-1, :] - generated_matrix[:, :img_height-1, 1:, :])

    return K.sum(K.pow(a + b, 1.25))

#calculates total loss
def all_loss(c_layer, s_layers, generated):
    #get weights
    cont_weight = args.cont_weight
    style_weight = args.style_weight
    variation_weight = args.var_weight

    #calculate content loss
    cont_features = c_layer[CONT_IMG_POS, :, :, :]
    generated_features = c_layer[RESULT_IMG_POS, :, :, :]
    c_loss = cont_loss(cont_features, generated_features)

    #calculate style loss
    s_loss = None
    for layer in s_layers:
        style_features = layer[STYLE_IMG_POS, :, :, :]
        generated_features = layer[RESULT_IMG_POS, :, :, :]
        if s_loss is None:
            s_loss = style_loss(style_features, generated_features) * (style_weight / len(s_layers))
        else:
            s_loss += style_loss(style_features, generated_features) * (style_weight / len(s_layers))

    #calculate variation loss 
    v_loss = variation_loss(generated)

    return cont_weight * c_loss + s_loss + variation_weight * v_loss

#computes losses and gradients
def compute_loss_and_grads(generated):

    generated = generated.reshape((1, img_height, img_width, 3))
    outs = f_outputs([generated])
    loss_value = outs[0]
    gradient_values = outs[1].flatten().astype('float64')
    return loss_value, gradient_values

#to save result Image
def save_image(filename, generated):

    # Reshape image and turn to RGB
    generated = generated.reshape((img_height, img_width, 3))
    generated = generated[:, :, ::-1]

    # Do shifts
    generated[:, :, 0] += 103.939
    generated[:, :, 1] += 116.779
    generated[:, :, 2] += 123.68

    # Clip values 0-255
    generated = np.clip(generated, 0, 255).astype('uint8')

    imsave(filename, Image.fromarray(generated))

#class for gradients and loss values 
class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.gradient_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, gradient_values = compute_loss_and_grads(x)
        self.loss_value = loss_value
        self.gradient_values = gradient_values
        return self.loss_value

    def gradients(self, x):
        assert self.loss_value is not None
        gradient_values = np.copy(self.gradient_values)
        self.loss_value = None
        self.gradient_values = None
        return gradient_values


if __name__ == '__main__':
    # Prepare the generated image
    result_img = np.random.uniform(0, 255, (1, img_height, img_width, 3)) - 128.

    # Load the content and style images
    cont = process_imgage(cont_path)
    style = process_imgage(style_path)

    # Prepare the variables for graph
    cont_image = K.variable(cont)
    style_image = K.variable(style)
    result_image = K.placeholder((1, img_height, img_width, 3))
    loss = K.variable(0.)

    # Get  layers to calculate the loss metric
    content_layer, style_layers = get_layers(cont_image, style_image, result_image)

    # Define loss and gradient
    loss = all_loss(content_layer, style_layers, result_image)
    gradients = K.gradients(loss, result_image)

    # Define the output
    outputs = [loss]
    outputs += gradients
    f_outputs = K.function([result_image], outputs)

    evaluator = Evaluator()
    iterations = args.iterations

    name = '{}-{}{}'.format(result_path, 0, result_ext)
    save_image(name, result_img)

    for i in range(iterations):
        print('Iteration:', i)
        result_img, min_val, info = fmin_l_bfgs_b(evaluator.loss, result_img.flatten(),
                                                     fprime=evaluator.gradients, maxfun=20)
        print('Loss:', min_val)
        name = '{}-{}{}'.format(result_path, i+1, result_ext)
        save_image(name, result_img)
        print('Saved image to: {}'.format(name))