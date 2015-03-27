import numpy
import time
import sys
import subprocess
import os
import random
import theano

from hyperParams import *
from load import loadFromImg
from load import loadOpticalFlow

# import rnn
from rnn import rnn

numpy.random.seed(42)
random.seed(423)
seed = 42

# HYPER PARAMETERS 
lr = 1e-2 # learning rate
save_epoch = 100
models_root = 'models/'
batch_size = 2
hidden_size = 2

# LOAD DATA
train_features_numpy, test_features_numpy, numtrain, numtest, \
data_mean, data_std = loadFromImg()
# train_features_theano = theano.shared(train_features_numpy)

train_ofx, train_ofy, test_ofx,  test_ofy = loadOpticalFlow()
train_labels = numpy.concatenate((train_ofx, train_ofy), axis = 1)
test_labels = numpy.concatenate((test_ofx, test_ofy), axis = 1)

train_rawframes = train_features_numpy
test_rawframes = test_features_numpy

# RESHAPE
# train_rawframes = \
# numpy.reshape(train_features_numpy, (trainframes, frame_len))
# test_rawframes = \
# numpy.reshape(test_features_numpy, (testframes, frame_len))
# 
# train_labels = numpy.reshape(train_labels, (trainframes, frame_len*2))
# test_labels = numpy.reshape(test_labels, (testframes, frame_len*2))

# print 'shapes of data: '
# print train_rawframes.shape
# print train_ofx.shape
# print train_ofy.shape
# print train_labels.shape

# INITIALIZATION
model = rnn(frame_len, frame_len*2, hidden_size, batch_size, numframes)

i = 0;
vinframes = train_rawframes[i*batch_size:(i+1)*batch_size, :]
vtruth = train_labels[i*batch_size:(i+1)*batch_size, :]
# cost = model.train(inframes, truth, lr)
