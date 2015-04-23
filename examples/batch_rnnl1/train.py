import numpy
import time
import sys
# import subprocess
# import os
import random
import theano
from time import clock

from hyper_params import *
sys.path.insert(0, project_path)
from utils.load import *
from utils.log import *

from rnn.batch_rnnl1 import Batch_RNNL1

seed = 42
numpy.random.seed(seed)
random.seed(seed)

logInfo = LogInfo('LOG.txt')
logInfo.mark('model: ')

# INITIALIZATION
tic = clock()
model = Batch_RNNL1(frame_dim, frame_dim*2, hidden_size, frame_dim, seq_len)
toc = clock()
logInfo.mark('time of initializing the model: ' + str(toc - tic))


# LOAD DATA
tic = clock()

features_train_numpy = \
  loadFrames(data_path + 'train/', image_shape, numframes_train, seq_len)
ofx_train, ofy_train = \
  loadOFs(data_path + 'train/', image_shape, numframes_train, seq_len)
labels_train = numpy.concatenate((ofx_train, ofy_train), axis = 1)

features_test_numpy = \
  loadFrames(data_path + 'test/', image_shape, numframes_test, seq_len)
ofx_test, ofy_test = \
  loadOFs(data_path + 'test/', image_shape, numframes_test, seq_len)
labels_test = numpy.concatenate((ofx_test, ofy_test), axis = 1)

# PREPROCESS
data_mean = features_train_numpy.mean()
data_std = features_train_numpy.std()
features_train_numpy -= data_mean
features_train_numpy /= data_std 
# features_train_numpy = features_train_numpy[numpy.random.permutation(numseqs_train)]
train_features_theano = theano.shared(features_train_numpy)
features_test_numpy -= data_mean
features_test_numpy /= data_std
test_feature_beginnings = features_test_numpy[:,:frame_dim*3]

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


# RESHAPE
rawframes_train = \
numpy.reshape(features_train_numpy, (numseqs_train, seq_dim))
labels_train = numpy.reshape(labels_train, (numseqs_train, seq_dim*2))

rawframes_test = \
numpy.reshape(features_test_numpy, (numseqs_test, seq_dim))
labels_test = numpy.reshape(labels_test, (numseqs_test, seq_dim*2))

toc = clock()
logInfo.mark('time of loading data: ' + str(toc - tic))



# TRAINING
preds_train = numpy.zeros(labels_train.shape)
preds_test = numpy.zeros(labels_test.shape)

squared_mean_train = numpy.mean(labels_train[1:, :] ** 2)
squared_mean_test = numpy.mean(labels_test[1:, :] ** 2)
logInfo.mark('squared_mean_train: ' + str(squared_mean_train))
logInfo.mark('squared_mean_test: ' + str(squared_mean_test))

epoch = 0
prev_cost = 1e10
decay = max_decay
while (1):
  epoch += 1
  # SHUFFLE
  [rawframes_train, labels_train] \
    = shuffle(rawframes_train, labels_train, 1, seed, en_shuffle)


  # TRAIN PHASE
  tic = clock()
  cost_train = 0.
  for i in xrange(num_batch_train):
    rows = range(i*batch_size_train, (i+1)*batch_size_train)
    cost_train+=model.train(rawframes_train[rows, :], labels_train[rows, :], lr)
  cost_train /= num_batch_train
  cost_train /= squared_mean_train
  toc = clock()


  cost_test = 0.
  for i in xrange(num_batch_test):
    rows = range(i*batch_size_test, (i+1)*batch_size_test)
    cost_test += model.getCost(rawframes_test[rows, :], labels_test[rows, :])
  cost_test /= num_batch_test 
  cost_test /= squared_mean_test

  cur_cost = cost_test

  logInfo.mark('# epoch: ' + str(epoch) \
      + '\tcost_train: ' + str(cost_train) + '\tcost_test: ' + str(cost_test) \
      +'\tlearning_rate: ' + str(lr) + '\ttime: ' + str(toc-tic))


  # LEARNING RATE DECAY
  if (en_decay and epoch >= pretrain_epoch):
    if prev_cost - cur_cost <= epsl_decay:  # decay
      if decay <= 0:
        lr /= 2
        decay = max_decay
        logInfo.mark('learning_rate decay to ' + str(lr))
      else :
        decay -= 1
    else:
      decay = max_decay
    

  # VALIDATE PHASE
  if (en_validate and epoch >= pretrain_epoch):
    if prev_cost <= cur_cost + epsl_validate: # reload
      model.load(models_path + 'model.npy')
      cur_cost = prev_cost
      logInfo.mark('load model ...')
    else: 
      # prev_cost = cur_cost
      model.save(models_path + 'model')

  if (epoch >= pretrain_epoch):
    prev_cost = cur_cost   # also used by learning rate decay


  # SAVE MODEL
  if (epoch % save_epoch == 0):
    model.save(models_path + 'model')

    # predictions
    for i in xrange(num_batch_train):
      rows = range(i*batch_size_train, (i+1)*batch_size_train)
      preds_train[rows, :] = model.predict(rawframes_train[rows, :])

    for i in xrange(num_batch_test):
      rows = range(i*batch_size_test, (i+1)*batch_size_test)
      preds_test[rows, :] = model.predict(rawframes_test[rows, :])

    numpy.save(pred_path + 'preds_train', preds_train)
    numpy.save(pred_path + 'preds_test', preds_test)

    logInfo.mark('saved model @ ' + models_path + 'model.npy')


  # BACKUP MODEL
  if (epoch % backup_epoch == 0):
    model.save(models_path + 'model_' + str(epoch))
    # numpy.save(pred_path + 'preds_train_' + str(epoch), preds_train)
    # numpy.save(pred_path + 'preds_test_' + str(epoch), preds_test)

    logInfo.mark('backuped model @ ' + models_path + 'model_' + str(epoch) + '.npy')

