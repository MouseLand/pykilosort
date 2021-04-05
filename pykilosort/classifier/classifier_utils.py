#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 22:46:50 2021

@author: rajat
"""

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib import rcParams 
rcParams['font.size'] = 20
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
rcParams['figure.autolayout'] = True
import warnings
warnings.filterwarnings("ignore")

# dt = maximum duration in the cross-correlogram
# st1, st2 = spike train
def compute_correlogram(st1, st2, dt=3.):
    spikediff = np.array([])
    T = (max(max(st1), max(st2))-min(min(st1), min(st2))); # total time range
    # we traverse both spike trains together, keeping track of the spikes in the first
    # spike train that are within dt of spikes in the second spike train
    ilow = 0; # lower bound index
    ihigh = 0; # higher bound index
    j = 0; # index of the considered spike
    while j<len(st2): # traverse all spikes in the second spike train
        while (ihigh<len(st1)) and (st1[ihigh] < st2[j]+dt):
            ihigh = ihigh + 1; 
        while (ilow<len(st1)) and (st1[ilow] < st2[j]-dt):
            ilow = ilow + 1; 
        if ilow>len(st1):
            break 
        if st1[ilow] > st2[j]+dt:
            # if the lower bound is actually outside of dt range, means we overshot (there were no spikes in range)
            # simply move on to next spike from second spike train
            j = j+1;
            continue;
        for k in range(ilow,ihigh):
            #for all spikes within plus/minus dt range
            diffval = st2[j] - st1[k]; 
            spikediff = np.append(spikediff,diffval)
        j = j+1;
    spikediff = spikediff[spikediff>=0]
    return spikediff

# st1, st2 = spike train
def get_logBinned_autocorr(st1,st2,dt=3,nbins=50):
    # -3.25 value is a hack to get as close to 0 as possible
    bins = np.logspace(-3.25,np.log10(dt),nbins)
    diffspike = compute_correlogram(st1,st2,dt)
    acorr, binedges = np.histogram(diffspike,bins)# log-spaced histogram
    # take difference between edges and divide the count 
    bedges_ = np.diff(binedges)
    acorr = acorr/bedges_
    # normalize the entire data by baseline count
    if np.nanmean(acorr)!=0:
        acorr = acorr/np.nanmean(acorr)
    else:
        acorr = np.array([0.0]*len(bedges_))
    return acorr

#calculate the confusion matrix
def plot_confusion_matrix(y_pred, y_test):    
    confmat = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred).T
    confmat = confmat/confmat.sum()
    
    LABEL_NAMES = ['noise','good/mua']
    n_labels = len(LABEL_NAMES)
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(confmat, cmap=plt.cm.Blues)
    ax.set_xticks(np.arange(n_labels))
    ax.set_yticks(np.arange(n_labels))
    ax.set_xticklabels(LABEL_NAMES)
    ax.set_yticklabels(LABEL_NAMES)
    # rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    # loop over data dimensions and create text annotations.
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(x=i, y=j, s=round(confmat[i, j],2), va='center', ha='center')
    # avoid that the first and last row cut in half
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    plt.tight_layout()
    plt.show()
    return confmat
