#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 23:19:45 2021

@author: rajat
"""

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, BatchNormalization, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation


# build ACG classifier
def buildACGmodel(inputShape=49,activation="relu", activationop='softmax'):
    # deep linear neural network
    model = Sequential()
    model.add(Dense(48, activation = activation, kernel_regularizer='l2', 
                    kernel_initializer = "he_normal",input_dim=inputShape,name='dense_101'))
    model.add(BatchNormalization(name='batchnorm_101'))
    model.add(Dense(24, activation=activation, kernel_regularizer='l2', 
                    kernel_initializer = "he_normal",name='dense_102'))
    model.add(BatchNormalization(name='batchnorm_103'))
    model.add(Dense(8, activation = activation, kernel_regularizer='l2', 
                    kernel_initializer = "he_normal",name='dense_103'))
    model.add(BatchNormalization(name='batchnorm_105'))
    model.add(Dense(2, activation=activationop,name='dense_104')) # 2 output classes
    # print(model.summary())
    return model

# compile ACG classifier
def fitACGmodel(model, xTrain, yTrain, xVal=None, yVal=None, batchSize=64, numEpochs=10):
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])   
    # fit the model
    if xVal is not None and yVal is not None:
        history = model.fit(x=xTrain,y=yTrain,batch_size=batchSize,
                        epochs=numEpochs, verbose=1, validation_data=(xVal, yVal))    
    else:
        history = model.fit(x=xTrain,y=yTrain,batch_size=batchSize,
                        epochs=numEpochs, verbose=1)
    return model, history

# predict using ACG classifier
def predictACGmodel(model, xTest):
    y_pred = model.predict_classes(xTest)
    return y_pred


# build WF based classifier
def buildWFmodel(inputShape=[32,82,1],activation="relu", activationop='softmax'):
    model = Sequential()
    # convolution block
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same',input_shape=inputShape))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Activation(activation))
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same',input_shape=inputShape))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(BatchNormalization(momentum=0.1))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))
    model.add(Flatten())
    # fully connected layers
    model.add(Dense(16, activation=activation, kernel_regularizer='l1_l2'))
    model.add(Dense(8, activation=activation, kernel_regularizer='l1_l2'))
    model.add(Dense(2, activation=activationop)) # 2 classes
    # summarize the model
    #print(model.summary())
    return model

# data augmentation
def augmentWFmodel(xTrain):  
    datagen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 vertical_flip=True,
                                 )
    datagen.fit(xTrain)
    return datagen


# compile and fit WF based model
def fitWFmodel(model, datagen, xTrain, yTrain, xVal=None, yVal=None, batchSize=64, numEpochs=15):
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])  
    # fit the model
    if xVal is not None and yVal is not None:
        history = model.fit_generator(datagen.flow(xTrain, yTrain, batchSize),
                        epochs=numEpochs, verbose=1, validation_data=(xVal, yVal))
    else:
        history = model.fit_generator(datagen.flow(xTrain, yTrain, batchSize),
                        epochs=numEpochs, verbose=1)
    return model, history

# predict WF based model
def predictWFmodel(model, xTest):
    y_pred = model.predict_classes(xTest)
    return y_pred


# Merged classifier

