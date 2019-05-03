from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from scipy.misc import imresize
import matplotlib.pyplot as plt
import tensorflow as tf
from ops import *

'''class Discriminator:
    def __init__(self, img_shape):

        self.img_rows, self.img_cols, self.channels = img_shape
        with tf.variable_scope('d'):
            print("Initializing discriminator weights")
            self.W1 = init_weights([5, 5, self.channels, 64])
            self.b1 = init_bias([64])
            self.W2 = init_weights([3, 3, 64, 64])
            self.b2 = init_bias([64])
            self.W3 = init_weights([3, 3, 64, 128])
            self.b3 = init_bias([128])
            self.W4 = init_weights([2, 2, 128, 256])
            self.b4 = init_bias([256])
            self.W5 = init_weights([7*7*256, 1])
            self.b5 = init_bias([1])'''

def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='C:/Users/Admin/Thesis/GAN/CNN/model', type=str, help='Path to model')
    parser.add_argument('-i', '--image', default='C:/Users/Admin/Thesis/NeuralStyleTransfer/KatsePildid/Barac_ObamaC.jpg', type=str, required=False, help='Path to input image')
    parser.add_argument('-s', '--size', type=int, default=28, help='Width x height for resizing image (square), '
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


def load_image(args):
    image = cv2.imread(args['image'])
    image = img_to_array(image)
    image = mk_square(image)
    image = cv2.resize(image, (args['size'], args['size']))
    image = image.astype('float') / 255.0
    plt.imshow(image)
    image = np.expand_dims(image, axis=0)
    return image


    def load_original_image(args):
        image = cv2.imread(args['image'])
        return image.copy()


    def load_trained_model(args):
        print('[INFO] loading model...')
        model = load_model(args['model'])
        return model


    def classify_image(model, image):
        (caricature, portrait) = model.predict(image)[0]

        label = 'Karikatuur' if caricature > portrait else 'Portree'
        print(label)
        probability = caricature if caricature > portrait else portrait
        print('Probability: ' + str(probability))
        label = '{}: {:.2f}%'.format(label, probability * 100)
        return label


    def __main__():
        args = load_arguments()
        image = load_image(args)
        model = load_trained_model(args)
        label = classify_image(model, image)

        orig = load_original_image(args)
        output = imutils.resize(orig, width=400)
        cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Image with predicted label', output)
        cv2.imwrite('../dataset/1.jpg', output)
        cv2.waitKey(0)


    if __name__ == "__main__":
        __main__()