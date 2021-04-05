#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 10:03:39 2021

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
import numpy as np
from classifier import *
from classifier_utils import *

# load testing data
xWFTrain = np.load('xWFTrain.npy', allow_pickle=True)
xACGTrain = np.load('xACGTrain.npy', allow_pickle=True)
yTrain = np.load('yTrain.npy', allow_pickle=True)
xWFVal = np.load('xWFVal.npy', allow_pickle=True)
xACGVal = np.load('xACGVal.npy', allow_pickle=True)
yVal = np.load('yVal.npy', allow_pickle=True)
xWFTest = np.load('xWFTest.npy', allow_pickle=True)
xACGTest = np.load('xACGTest.npy', allow_pickle=True)
yTest = np.load('yTest.npy', allow_pickle=True)

# test autocorrelogram model compilation
acgmodel = build_ACGmodel(inputShape=xACGTrain.shape[1])
acgmodel, historyacg = fit_ACGmodel(acgmodel, xACGTrain, yTrain, xACGVal, yVal)
y_pred = acgmodel.predict_classes(xACGTest)
plot_confusion_matrix(y_pred, yTest)

# load autocorrelogram model
