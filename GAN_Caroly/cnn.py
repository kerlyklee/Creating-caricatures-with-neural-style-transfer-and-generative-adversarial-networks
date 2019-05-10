#!/usr/bin/env python3
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dense
from keras import backend as K
from keras.utils import plot_model


class CNN:

    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()

        input_shape = (height, width, depth)

        if K.image_data_format() == 'channels_first':
            input_shape = (depth, height, width)

        model.add(Convolution2D(20, (5, 5), padding='same', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Convolution2D(50, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())

        model.add(Dense(500))
        model.add(Activation('relu'))
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        print(model.summary())

        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        return model
