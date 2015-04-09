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
features_train_numpy, features_test_numpy, train_numseqs, numseqs_test, \
data_mean, data_std = loadFromImg()
# train_features_theano = theano.shared(features_train_numpy)

ofx_train, ofy_train, ofx_test,  ofy_test = loadOpticalFlow()
train_labels = numpy.concatenate((ofx_train, ofy_train), axis = 1)
test_labels = numpy.concatenate((ofx_test, ofy_test), axis = 1)

train_rawframes = features_train_numpy
test_rawframes = features_test_numpy

# RESHAPE
# train_rawframes = \
# numpy.reshape(features_train_numpy, (numframes_train, frame_dim))
# test_rawframes = \
# numpy.reshape(features_test_numpy, (numframes_test, frame_dim))
# 
# train_labels = numpy.reshape(train_labels, (numframes_train, frame_dim*2))
# test_labels = numpy.reshape(test_labels, (numframes_test, frame_dim*2))

# print 'shapes of data: '
# print train_rawframes.shape
# print ofx_train.shape
# print ofy_train.shape
# print train_labels.shape

# INITIALIZATION
model = rnn(frame_dim, frame_dim*2, hidden_size, batch_size, seq_len)

i = 0;
vinframes = train_rawframes[i*batch_size:(i+1)*batch_size, :]
vtruth = train_labels[i*batch_size:(i+1)*batch_size, :]
# cost = model.train(inframes, truth, lr)
