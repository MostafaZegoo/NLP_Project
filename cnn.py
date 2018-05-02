# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 21:02:50 2018

@author: CS
"""

import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow as tf
from keras import backend as k
import scipy.io
import pandas as pd
from sklearn.model_selection import train_test_split


def categorical_hinge(y_true, y_pred):
    pos = k.sum(y_true * y_pred, axis=-1)
    neg = k.max((1.0 - y_true) * y_pred, axis=-1)
    return k.mean(k.maximum(0.0, neg - pos + 1), axis=-1)


def perpare_data_Set():
    matContent = pd.read_csv('uci-news-aggregator.csv')
    value = list(matContent.values())

    value = value[3:]
    list_image = []
    list_label = []

    for val in value:
        for val_ in val:
            list_image.append(val_[0])
            list_label.append(val_[1])

    """
    print (list_label)
    print ("*******************")
    print (list_image)"""
    """
    label=[]
    for val in list_label:
        label.append(val[0])

    print (label)

    image=[]
    for val in list_image:
        image.append(val[0])
    print (image)
    print(len(image),len(label))
    """
    X = np.array(list_image)
    y = np.array(list_label)
    return X, y


# Data Preparing

batch_size = 4
nr_classes = 14
nr_iterations = 100
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# X_train = X_train.reshape(60000, 784)
# X_test = X_test.reshape(10000, 784)
"""X_train = X_train.reshape(1130, 40000)

X_test = X_test.reshape(283, 40000)"""

X, y = perpare_data_Set()

X_train = X[0:1130, :]
y_train = y[0:1130, :]

X_test = X[1130:, :]
y_test = y[1130:, :]

X_train = np.expand_dims(X_train, 3)
X_test = np.expand_dims(X_test, 3)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(y_train, nr_classes)
Y_test = np_utils.to_categorical(y_test, nr_classes)

input_shape = (200, 200, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(14, activation='softmax'))

X_val = X_train[0:226, :]
Y_val = Y_train[0:226, :]

X_train = X_train[226:1130, :]
Y_train = Y_train[226:1130, :]

model.summary()
model.compile(loss='mse',
              optimizer='sgd',
              metrics=['accuracy'])

saved_weights_name = 'SVMWeights.h5'

checkpoint = ModelCheckpoint(saved_weights_name,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True,
                             mode='max')
csv_logger = CSVLogger('v.csv')

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nr_iterations,
                    verbose=1, validation_data=(X_val, Y_val), callbacks=[checkpoint, csv_logger])

score = model.evaluate(X_test, Y_test, verbose=0)
