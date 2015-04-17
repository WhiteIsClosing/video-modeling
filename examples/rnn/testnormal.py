
import numpy
import time
import sys
import subprocess
import os
import random
import theano
from time import clock

from hyperParams import *
from load import loadFrames
from load import loadOpticalFlow
from rnn import RNN
from utils import LogInfo

seed = 42
numpy.random.seed(seed)
random.seed(423)

# LOAD DATA
features_train_numpy, features_test_numpy = loadFrames()

ofx_train, ofy_train, ofx_test,  ofy_test = loadOpticalFlow()
labels_train = numpy.concatenate((ofx_train, ofy_train), axis = 1)
labels_test = numpy.concatenate((ofx_test, ofy_test), axis = 1)


# PREPROCESS
# Normalize 
data_mean = features_train_numpy.mean()
data_std = features_train_numpy.std()
features_train_numpy -= data_mean
features_train_numpy /= data_std 
# features_train_numpy = features_train_numpy[numpy.random.permutation(numseqs_train)]
train_features_theano = theano.shared(features_train_numpy)
features_test_numpy -= data_mean
features_test_numpy /= data_std
test_feature_beginnings = features_test_numpy[:,:frame_dim*3]

# Normalize 
ofx_mean = ofx_train.mean()
ofx_std = ofx_train.std()
ofx_train -= ofx_mean
ofx_train /= ofx_std
ofx_test -= ofx_mean
ofx_test /= ofx_std

ofy_mean = ofy_train.mean()
ofy_std = ofy_train.std()
ofy_train -= ofy_mean
ofy_train /= ofy_std
ofy_test -= ofy_mean
ofy_test /= ofy_std

print data_mean
print data_std
print ofx_mean
print ofx_std
print ofy_mean
print ofy_std

print numpy.max(features_train_numpy)
print numpy.max(ofx_train)
print numpy.max(ofy_train)
print numpy.min(features_train_numpy)
print numpy.min(ofx_train)
print numpy.min(ofy_train)
