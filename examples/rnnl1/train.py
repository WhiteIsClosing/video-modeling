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

from rnn.rnnl1 import RNNL1

seed = 42
numpy.random.seed(seed)
random.seed(seed)

logInfo = LogInfo('LOG.txt')
logInfo.mark('model: ')

# INITIALIZATION
tic = clock()
model = RNNL1(frame_dim, frame_dim*2, hidden_size)
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
numpy.reshape(features_train_numpy, (numframes_train, frame_dim))
labels_train = numpy.reshape(labels_train, (numframes_train, frame_dim*2))

rawframes_test = \
numpy.reshape(features_test_numpy, (numframes_test, frame_dim))
labels_test = numpy.reshape(labels_test, (numframes_test, frame_dim*2))

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
    [rawframes_train, labels_train] =\
         shuffle(rawframes_train, labels_train, seq_len, seed, en_shuffle)


    # TRAIN PHASE
    tic = clock()
    cost_train = 0.
    for i in xrange(numseqs_train):
        rows = range(i*seq_len, (i+1)*seq_len)
        cost_train +=\
             model.train(rawframes_train[rows, :], labels_train[rows, :], lr)
    cost_train /= numseqs_train
    cost_train /= squared_mean_train
    toc = clock()


    cost_test = 0.
    for i in xrange(numseqs_test):
        rows = range(i*seq_len, (i+1)*seq_len)
        cost_test += model.getCost(rawframes_test[rows, :],labels_test[rows, :])
    cost_test /= numseqs_test 
    cost_test /= squared_mean_test

    cur_cost = cost_test

    logInfo.mark('# epoch: ' + str(epoch) \
        + '\tcost_train: ' + str(cost_train) + '\tcost_test: ' + str(cost_test)\
        +'\tlearning_rate: ' + str(lr) + '\ttime: ' + str(toc-tic))


    # LEARNING RATE DECAY
    if (epoch >= pretrain_epoch):
        if (en_decay):
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
        if (en_validate):
            if prev_cost <= cur_cost + epsl_validate: # reload
                cur_cost = prev_cost
                model.load(models_path + 'model.npy')
                logInfo.mark('load model ...')
            else: 
                model.save(models_path + 'model')

        if prev_cost <= cur_cost: 
            cur_cost = prev_cost

        prev_cost = cur_cost   # also used by learning rate decay


    # SAVE MODEL
    if (epoch % save_epoch == 0):
        model.save(models_path + 'model')

        # predictions
        for i in xrange(numseqs_train):
            rows = range(i*seq_len, (i+1)*seq_len)
            preds_train[rows, :] = model.predict(rawframes_train[rows, :])

        for i in xrange(numseqs_test/seq_len):
            rows = range(i*seq_len, (i+1)*seq_len)
            preds_test[rows, :] = model.predict(rawframes_test[rows, :])

        numpy.save(pred_path + 'preds_train', preds_train)
        numpy.save(pred_path + 'preds_test', preds_test)

        logInfo.mark('saved model @ ' + models_path + 'model.npy')


    # BACKUP MODEL
    if (epoch % backup_epoch == 0):
        model.save(models_path + 'model_' + str(epoch))
        # numpy.save(pred_path + 'preds_train_' + str(epoch), preds_train)
        # numpy.save(pred_path + 'preds_test_' + str(epoch), preds_test)

        logInfo.mark('backuped model @ '+models_path+'model_'+str(epoch)+'.npy')

