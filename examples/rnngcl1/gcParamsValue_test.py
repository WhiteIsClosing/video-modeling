import pylab
from collections import OrderedDict
import numpy
import numpy.random
numpy_rng  = numpy.random.RandomState(1)
from scipy import misc

from hyperParams import *
from load import *
from logInfo import *
from plot import *
from gcParams import *
from gcParamsValue import *

gcParams = GCParams(numvis=frame_dim,
                    numnote=0,
                    numfac=numfac_,
                    numvel=numvel_,
                    numvelfac=numvelfac_,
                    numacc=numacc_,
                    numaccfac=numaccfac_,
                    numjerk=numjerk_,
                    seq_len_to_train=seq_len_to_train_,
                    seq_len_to_predict=seq_len_to_predict_,
                    output_type='real',
                    vis_corruption_type='zeromask',
                    vis_corruption_level=0.0,
                    numpy_rng=numpy_rng,
                    theano_rng=theano_rng)
gcParams.load(gc_path + 'model.npy')

gc = GCParamsValue(gcParams)

# print gcParams.wxf_left.get_value()
# print gcParamsValue.wxf_left

print 'loading data ...'

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

print '... done'

vels = gc.getVels(rawframes_train)
print vels.shape
print vels
