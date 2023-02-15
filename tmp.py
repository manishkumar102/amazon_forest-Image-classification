import numpy as np
import pandas as pd
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import fbeta_score

from PIL import Image

import os
import cfg


def load_model():
    #Use VGG16 imagenet weights as base model
    input_tensor = Input(shape=cfg.img_dim)
    base_model = VGG16(include_top=False,
                           weights='imagenet',
                           input_shape=cfg.img_dim)
    bn = BatchNormalization()(input_tensor)
    x = base_model(bn)
    x = Flatten()(x)
    output = Dense(cfg.num_classes, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    #loading the weights that we trained
    model.load_weights('notebooks/weights/weights.best.hdf5')

    return model

model = load_model()

def preprocess_image(filename):
    img = Image.open(filename)
    img.thumbnail((128, 128))

    # Convert to RGB and normalize
    img_array = np.asarray(img.convert("RGB"), dtype=np.float32)

    img_array = img_array[:, :, ::-1]
    # Zero-center by mean pixel
    img_array[:, :, 0] -= 103.939
    img_array[:, :, 1] -= 116.779
    img_array[:, :, 2] -= 123.68

    img_array = img_array / 255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predictClass(model, img_array):
    yhat = model.predict(img_array)
    # This We use 0 because we are only predicting one image
    # This could br easily extended to multiple images
    labels = [cfg.ymap[i] for i, value in enumerate(yhat[0]) if value > cfg.thresholds[i]]

    return labels


filepath = './input/test-jpg/test_0.jpg'
img_array = preprocess_image(filepath)

labels = predictClass(model, img_array)

print(labels)
