import matplotlib
matplotlib.use('Agg') # This line must be put before any matplotlib imports 
import matplotlib.pyplot as plt

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
from pgp3layerParams import *


max_num = 100

# LOAD PREDICTION
# idx = 1
# pred_frames_train = numpy.load(pred_path + 'pred_frames_train_' + str(idx) + '.npy')
# pred_frames_test = numpy.load(pred_path + 'pred_frames_test_' + str(idx) + '.npy')

pred_frames_train = numpy.load(pred_path + 'pred_frames_train.npy')
pred_frames_test = numpy.load(pred_path + 'pred_frames_test.npy')

features_train_numpy, features_test_numpy = loadFrames()

true_frames_train = features_train_numpy
true_frames_test = features_test_numpy

# # INVERSE PREPROCESS
# print 'preprocessing ...'
# # Normalize 
# data_mean = features_train_numpy.mean()
# data_std = features_train_numpy.std()

# LOAD MODEL PARAMETERS
modelParams = Pgp3layerParams(numvis=frame_dim,
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

modelParams.load(models_path + 'model.npy')
print modelParams.wxf_left.get_value().shape
print modelParams.wxf_right.get_value().shape


print '... done'

print 'ploting images ...'

path_train = vis_path + 'train/'
plotFrames(true_frames_train, image_shape, path_train + 'true_frames/', max_num)
plotFrames(pred_frames_train, image_shape, path_train + 'pred_frames/', max_num)
path_test = vis_path + 'test/'
plotFrames(true_frames_test, image_shape, path_test + 'true_frames/', max_num)
plotFrames(pred_frames_test, image_shape, path_test + 'pred_frames/', max_num)

path_u = vis_path + 'u/'
wxf_left = modelParams.wxf_left.get_value().T.flatten()
plotFrames(wxf_left, image_shape, path_u, max_num)
path_u = vis_path + 'v/'
wxf_right = modelParams.wxf_right.get_value().T.flatten()
plotFrames(wxf_right, image_shape, path_u, max_num)


print '... done'

