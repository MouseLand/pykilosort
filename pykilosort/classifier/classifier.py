#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 23:19:45 2021

@author: rajat
"""

import numpy as np
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
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# build ACG classifier
def build_ACGmodel(inputShape=49,activation="relu", activationop='softmax'):
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
def fit_ACGmodel(model, xTrain, yTrain, xVal=None, yVal=None, 
                 batchSize=64, numEpochs=10):
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
def predict_ACGmodel(model, xTest):
    y_pred = model.predict_classes(xTest)
    return y_pred


# build WF based classifier
def build_WFmodel(inputShape=[32,82,1], activation="relu", activationop='softmax'):
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
def augment_WFmodel(xTrain):  
    datagen = ImageDataGenerator(width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 vertical_flip=True,
                                 )
    datagen.fit(xTrain)
    return datagen


# compile and fit WF based model
def fit_WFmodel(model, datagen, xTrain, yTrain, xVal=None, yVal=None, 
                batchSize=64, numEpochs=15):
    # compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])  
    # fit the model
    if xVal is not None and yVal is not None:
        history = model.fit(datagen.flow(xTrain, yTrain, batchSize),
                        epochs=numEpochs, verbose=1, validation_data=(xVal, yVal))
    else:
        history = model.fit(datagen.flow(xTrain, yTrain, batchSize),
                        epochs=numEpochs, verbose=1)
    return model, history

# predict WF based model
def predict_WFmodel(model, xTest):
    y_pred = model.predict_classes(xTest)
    return y_pred


# Multiplication based ensemble model classifier
def build_ensembleModel(acgmodel_fname='modelACG.h5', wfmodel_fname='modelWF.h5', 
                        ensembleMode='multiply'):
    acg_model = tf.keras.models.load_model(acgmodel_fname)
    wf_model = tf.keras.models.load_model(wfmodel_fname)
    models = [wf_model, acg_model]
    outputs = []
    for model in models:
        outputs.append(model.outputs[0])
    if ensembleMode=='average':
        ymod = tf.keras.layers.Average()(outputs)
    elif ensembleMode=='add':
        ymod = tf.keras.layers.Add()(outputs)
    elif ensembleMode=='multiply':
        ymod = tf.keras.layers.Multiply()(outputs)
    ensmodel = Model([model.input for model in models], ymod, name='ensemble')
    return ensmodel



# Integrated classifier built using WF and ACG classifier
def build_IntegratedModel(acgmodel_fname='modelACG.h5', wfmodel_fname='modelWF.h5', 
                          layerTrainable=False):
    acg_model = tf.keras.models.load_model(acgmodel_fname)
    wf_model = tf.keras.models.load_model(wfmodel_fname)
    models = [wf_model, acg_model]
    # update all layers in all models to not be trainable
    for i in range(len(models)):
        model = models[i]
        model.pop()
        for layer in model.layers:
            # make not trainable
            layer.trainable = layerTrainable
            # rename to avoid 'unique layer name' issue
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in models]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in models]
    #print(ensemble_outputs)
    merge = concatenate(ensemble_outputs)
    hidden1 = Dense(16, activation='relu')(merge)
    hidden2 = Dense(8, activation='relu')(hidden1)
    output = Dense(2, activation='softmax')(hidden2)
    model = Model(inputs=ensemble_visible, outputs=output)
    return model

# fit the integrated model
def fit_IntegratedModel(ensModel, xWFTrain, xACGTrain, yTrain, 
                        xWFVal=None, xACGVal=None, yVal=None, 
                        batchSize=64, numEpochs=20):
    # compile
    ensModel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    # fit stacked model using the ensemble
    if xWFVal is not None and xACGVal is not None and yVal is not  None:
        ensModel.fit([xWFTrain, xACGTrain], yTrain, validation_data=([xWFVal, xACGVal], yVal), 
                   batch_size=batchSize, epochs=numEpochs, verbose=1)
    else:
        ensModel.fit([xWFTrain, xACGTrain], yTrain, batch_size=batchSize, 
                     epochs=numEpochs, verbose=1)
    return ensModel

# predict using ensemble based model
def predict_ensembleModel(ensmodel, xWFTest, xACGTest):
    y_pred = ensmodel.predict([xWFTest, xACGTest])
    y_pred = np.argmax(y_pred, 1)
    return y_pred
