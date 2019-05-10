
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from scipy.misc import imresize
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from imutils import paths
from cnn import CNN 
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import cv2
import os
import easydict
import sys; sys.argv=['']; del sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='C:/Users/Admin/Thesis/GAN_Caroly/model', type=str, help='Path to model')
    parser.add_argument('-d', '--dataset', default='C:/Users/Admin/Thesis/GAN_Caroly/Data', type=str, help='Path to dataset')
    parser.add_argument('-e', '--epochs', required=False, type=int, default=25, help='Number of epochs')
    parser.add_argument('-ap', '--acc_plot', type=str, default='50_cnn_mk_square/acc_plot.png', help='Path to accuracy plot')
    parser.add_argument('-lp', '--loss_plot', type=str, default='50_cnn_mk_square/loss_plot.png', help='Path to loss plot')
    parser.add_argument('-s', '--size', type=int, default=28, help='Width x height number for resizing image (square)')
   # args = easydict.EasyDict({
    #    "model": 'C:/Users/Admin/Thesis/GAN/model' ,
     #   "dataset": 'C:/Users/Admin/Thesis/GAN/dataset_50',
        #"epochs": 25,
       # "acc_plot": 'acc_plot.png',
      #  "loss_plot": 'loss_plot.png',
        #"size": 28
   # })
    args = vars(parser.parse_args())


    return args


def acc_plot(args, epochs, history):
    plt.plot(np.arange(0, epochs), history.history['acc'], label='Training accuracy')
    plt.plot(np.arange(0, epochs), history.history['val_acc'], label='Validation accuracy')
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(args['acc_plot'])
    plt.clf()


def loss_plot(args, epochs, history):
    plt.plot(np.arange(0, epochs), history.history['loss'], label='Training loss')
    plt.plot(np.arange(0, epochs), history.history['val_loss'], label='Validation loss')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(args['loss_plot'])
    plt.clf()


def mk_square(img, desired_shp=1000):
    x, y, _ = img.shape
    maxs = max(img.shape[:2])
    y2 = (maxs-y)//2
    x2 = (maxs-x)//2
    arr = np.zeros((maxs, maxs, 3), dtype=np.float32)
    arr[int(np.floor(x2)):int(np.floor(x2)+x), int(np.floor(y2)):int(np.floor(y2)+y)] = img
    return imresize(arr, (desired_shp, desired_shp))


def __main__():
    args = load_arguments()
    epochs = int(args['epochs'])
    initial_learning_rate = 1e-3
    batch_size = 32

    print('[INFO] loading images...')

    images = []
    labels = []

    image_paths = sorted(list(paths.list_images(args['dataset'])))
    random.seed(42)
    random.shuffle(images)

    i = 0

    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = img_to_array(image)
        i = i + 1
        print('[INFO] reading in image %s...' % (str(i)))
        sq_big_image = mk_square(image)
        label = image_path.split(os.path.sep)[-2]
        sq_l_image = cv2.resize(sq_big_image, (args['size'], args['size']))
        label = 1 if label == 'caricature' else 0
        labels.append(label)
        images.append(sq_l_image)

    images = np.array(images, dtype='float') / 255.0
    labels = np.array(labels)

    (x_train, x_test, y_train, y_test) = train_test_split(images, labels, test_size=0.2, random_state=42)

    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)

    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode='nearest')

    print('[INFO] compiling model...')
    model = CNN().build(width=args['size'], height=args['size'], depth=3, classes=2)
    optimizer = Adam(lr=initial_learning_rate, decay=initial_learning_rate / epochs)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    print('[INFO] training network...')
    history = model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),
                                  validation_data=(x_test, y_test), steps_per_epoch=len(x_train) // batch_size,
                                  epochs=epochs, verbose=1)

    print('[INFO] serializing network...')
    model.save(args['model'])

    acc_plot(args, epochs, history)
    loss_plot(args, epochs, history)


if __name__ == '__main__':
    __main__()
