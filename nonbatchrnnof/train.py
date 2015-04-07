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

rawframes_train = features_train_numpy
rawframes_test = features_test_numpy

# RESHAPE
rawframes_train = \
numpy.reshape(features_train_numpy, (numframes_train, frame_dim))
labels_train = numpy.reshape(labels_train, (numframes_train, frame_dim*2))

rawframes_test = \
numpy.reshape(features_test_numpy, (numframes_test, frame_dim))
labels_test = numpy.reshape(labels_test, (numframes_test, frame_dim*2))

#LOG# 
logInfo = LogInfo('LOG.txt')

# INITIALIZATION
tic = clock()
model = RNN(frame_dim, frame_dim*2, hidden_size)
toc = clock()

#LOG#
logInfo.mark('initial time: ' + str(toc - tic))

'''
Data format:
frame[0]          ...   frame[seq_len-1]
frame[seq_len]  ...

...

frame[(numseqs_train-1)*seq_len]   ...   frame[]

'''

# TRAINING
preds_train = numpy.zeros(labels_train.shape)
preds_test = numpy.zeros(labels_test.shape)

epoch = 0
prev_cost = 1e10
decay = max_decay
while (1):
  epoch += 1

  # SHUFFLE
  numpy.random.seed(seed)
  numpy.random.shuffle(rawframes_train)
  numpy.random.shuffle(labels_train)

  # TRAIN PHASE
  tic = clock()
  cost_train = 0.
  for i in xrange(numseqs_train):
    cost_train += model.train(rawframes_train[i*seq_len:(i+1)*seq_len, :], labels_train[i*seq_len:(i+1)*seq_len, :], lr)
  cost_train /= (numseqs_train/seq_len)
  toc = clock()

  cost_test = 0.
  for i in xrange(numseqs_test/seq_len):
    cost_test += model.getCost(rawframes_test[i*seq_len:(i+1)*seq_len, :], labels_test[i*seq_len:(i+1)*seq_len, :])
  cost_test /= (numseqs_test/seq_len) 

  #LOG#
  logInfo.mark('# epoch: ' + str(epoch) + '\tcost_train: ' + str(cost_train) + '\tcost_test: ' + str(cost_test) +'\tlearning_rate: ' + str(lr) + '\ttime: ' + str(toc-tic))


  # VALIDATE PHASE
  if prev_cost <= cost_test:
    model.load(models_path + 'model.npy')

    #LOG#
    logInfo.mark('load model ...')

  else: 
    prev_cost = cost_test
    model.save(models_path + 'model')


  # LEARNING RATE DECAY
  if prev_cost - cost_test <= epsl:
    if decay > 0:
      decay -= 1
    else:
      lr /= 2
      decay = max_decay

      #LOG#
      logInfo.mark('learning_rate decay to ' + str(lr))

  else:
    decay = max_decay


  # SAVE MODEL
  if (epoch % save_epoch == 0):
    model.save(models_path + 'model_' + str(epoch))

    # predictions
    for i in xrange(numseqs_train):
      preds_train[i*seq_len:(i+1)*seq_len, :] = model.predict(rawframes_train[i*seq_len:(i+1)*seq_len, :])

    for i in xrange(numseqs_test/seq_len):
      preds_test[i*seq_len:(i+1)*seq_len, :] = model.predict(rawframes_test[i*seq_len:(i+1)*seq_len, :])

    numpy.save(preds_path + 'preds_train' + str(epoch), preds_train)
    numpy.save(preds_path + 'preds_test' + str(epoch), preds_test)

    #LOG#
    logInfo.mark('saved model @ ' + models_path + 'model_' + str(epoch))

