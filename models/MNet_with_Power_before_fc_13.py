# coding: UTF-8
#2018/12/27 by Jo Aoe
#Classify the (800ms*160ch) data to 2 diseases

#There is some redundant imports
#chainer_version = '5.0.0'
import numpy as np
import cupy
import chainer
from chainer import backend
from chainer import backends
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from chainer import iterators
from chainer import optimizers
import numpy as np
from chainer.dataset import concat_examples
from chainer.backends.cuda import to_cpu
import os
import argparse
import glob
import wave
import datetime
import itertools
import scipy.io
import h5py
from sklearn import utils
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
import random
import gc
import csv
import matplotlib.pyplot as plt
import pandas as pd
import random as rd
import math
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from multiprocessing import Pool
from multiprocessing import Process
from sklearn.metrics import accuracy_score
import sys

class MNet_with_Power_before_fc_13(Chain):
    def __init__(self):
        super(MNet_with_Power_before_fc_13, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(1, 32, (160, 64), stride=(1, 2))
            self.conv2 = L.Convolution2D(32, 64, (1, 16), stride=(1, 2))
            self.conv3 = L.Convolution2D(1, 32, (8, 8))
            self.conv4 = L.Convolution2D(32, 32, (8, 8))
            self.conv5 = L.Convolution2D(32, 64, (1, 4))
            self.conv6 = L.Convolution2D(64, 64, (1, 4))
            self.conv7 = L.Convolution2D(64, 128, (1, 2))
            self.conv8 = L.Convolution2D(128, 128, (1, 2))
            self.conv9 = L.Convolution2D(128, 256, (1, 2))
            self.conv10 = L.Convolution2D(256, 256, (1, 2))

            self.fc11 = L.Linear(None, 1024)
            self.norm1 = L.BatchNormalization(1024)
            self.fc12 = L.Linear(1024, 1024)
            self.norm2 = L.BatchNormalization(1024)
            self.fc13 = L.Linear(None, 2)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, (1, 2), stride=(1, 2))
        h = F.reshape(h, (h.data.shape[0], h.data.shape[2], h.data.shape[1], h.data.shape[3]))

        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, (5, 3))
        h = F.relu(self.conv5(h))
        h = F.relu(self.conv6(h))
        h = F.max_pooling_2d(h, (1, 2))
        h = F.relu(self.conv7(h))
        h = F.relu(self.conv8(h))
        h = F.max_pooling_2d(h, (1, 2))
        h = F.relu(self.conv9(h))
        h = F.relu(self.conv10(h))
        h = F.max_pooling_2d(h, (1, 2))

        #Get power spectrum by FFT. data_shape = (6,)
        # frequency by 1.25Hz
        # sampling_rate=1000Hz
        # freq_names = {'delta','theta','lalpha','halpha','beta','lgamma'};
        # freq_bands = [1 4; 4 8; 8 10; 10 13; 13 30; 30 50];
        # delta:1.25-3.75:
        # theta: 5-7.5:
        # lalpha:8.75-10:
        # halpha:11.25-12.5:
        # beta:13.75-30:
        # lgamma:31.25-50:
        tmp = cupy.abs(cupy.fft.fft(x))

        delta = cupy.average(tmp[:, :, :, 1:4], axis=3)
        theta = cupy.average(tmp[:, :, :, 4:7], axis=3)
        lalpha = cupy.average(tmp[:, :, :, 7:9], axis=3)
        halpha = cupy.average(tmp[:, :, :, 9:11], axis=3)
        beta = cupy.average(tmp[:, :, :, 11:25], axis=3)
        lgamma = cupy.average(tmp[:, :, :, 25:41], axis=3)
        Sum = delta + theta + lalpha + halpha + beta + lgamma

        power_spectral = cupy.zeros((x.shape[0],x.shape[1],x.shape[2], 6))
        power_spectral[:,:,:,0]= cupy.divide(delta,Sum)
        power_spectral[:, :, :,1] = cupy.divide(theta, Sum)
        power_spectral[:, :, :, 2] = cupy.divide(lalpha, Sum)
        power_spectral[:,:,:,3]= cupy.divide(halpha,Sum)
        power_spectral[:, :, :, 4] = cupy.divide(beta, Sum)
        power_spectral[:, :, :, 5] = cupy.divide(lgamma, Sum)
        power_spectral = chainer.Variable( power_spectral)
        power_spectral = F.cast(power_spectral, cupy.float32)

        h = F.reshape(h, (h.shape[0],h.shape[1]*h.shape[2]*h.shape[3]))
        power_spectral = F.reshape(power_spectral, (power_spectral.shape[0],power_spectral.shape[1]*power_spectral.shape[2]*power_spectral.shape[3]))

        h = F.relu(self.norm1(self.fc11(h)))
        h = F.dropout(h)
        h = F.relu(self.norm2(self.fc12(h)))
        h = F.dropout(h)
        #Concatenate the features extracted by deep neural network and relative power spectrum
        h = F.concat((h, power_spectral), axis=1)
        h = self.fc13(h)

        if chainer.config.train:
            return h
        return F.softmax(h)
