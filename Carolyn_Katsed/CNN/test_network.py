from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from scipy.misc import imresize
import matplotlib.pyplot as plt


def load_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Path to trained model')
    parser.add_argument('-i', '--image', required=True, help='Path to input image')
    parser.add_argument('-s', '--size', type=int, default=28, help='Width x height number for resizing image (square), '
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
    image = mk_square(image)
    image = cv2.resize(image, (args['size'], args['size']))
    image = image.astype('float') / 255.0
    image = img_to_array(image)
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

    label = 'Caricature' if caricature > portrait else 'Portrait'
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
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (1, 0, 0), 2)

    cv2.imshow('Image with predicted label', output)
    cv2.waitKey(0)


if __name__ == "__main__":
    __main__()
