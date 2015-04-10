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

idx = 3
max_num = 100
th_of = 0.1

features_train_numpy, features_test_numpy = loadFrames()
ofx_train, ofy_train, ofx_test, ofy_test = loadOpticalFlow()

preds_train = numpy.load(pred_path + 'preds_train_' + str(idx) + '.npy')
ofx_pred_train = preds_train[:, :frame_dim]
ofy_pred_train = preds_train[:, frame_dim:]
frames_train = features_train_numpy

preds_test = numpy.load(pred_path + 'preds_test_' + str(idx) + '.npy')
ofx_pred_test = preds_test[:, :frame_dim]
ofy_pred_test = preds_test[:, frame_dim:]
frames_test = features_test_numpy

# # PREPROCESS
# print 'preprocessing ...'
# # Normalize 
# data_mean = features_train_numpy.mean()
# data_std = features_train_numpy.std()
# 
# # Normalize 
# ofx_mean = ofx_train.mean()
# ofx_std = ofx_train.std()
# ofy_mean = ofy_train.mean()
# ofy_std = ofy_train.std()
# 
# frames_train = frames_train * data_std + data_mean
# ofx_train = ofx_train * ofx_std + ofx_mean
# ofx_pred_train = ofx_pred_train * ofx_std + ofx_mean
# ofy_train = ofy_train * ofy_std + ofy_mean
# ofy_pred_train = ofy_pred_train * ofy_std + ofy_mean

# frames_train = frames_train * 50
# ofx_train = ofx_train * 50
# ofx_pred_train = ofx_pred_train * 50
# ofy_train = ofy_train * 50
# ofy_pred_train = ofy_pred_train * 50

print '... done'


print 'ploting images ...'

path_train = vis_path + 'train/'
plotFrames(frames_train, image_shape, path_train + 'true_frames/', max_num)
plotOFs(ofx_train, ofy_train, th_of, 0., image_shape, \
            path_train + 'true_of/', max_num)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_num)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_num)
plotOFs(ofx_pred_train, ofy_pred_train, th_of, 0., image_shape, \
            path_train + 'pred_of/', max_num)
plotFrames(ofx_pred_train, image_shape, path_train + 'pred_ofx/', max_num)
plotFrames(ofy_pred_train, image_shape, path_train + 'pred_ofy/', max_num)


path_test = vis_path + 'test/'
plotFrames(frames_test, image_shape, path_test + 'true_frames/', max_num)
plotOFs(ofx_test, ofy_test, th_of, 0., image_shape, \
            path_test + 'true_of/', max_num)
plotFrames(ofx_test, image_shape, path_test + 'true_ofx/', max_num)
plotFrames(ofx_test, image_shape, path_test + 'true_ofx/', max_num)
plotOFs(ofx_pred_test, ofy_pred_test, th_of, 0., image_shape, \
            path_test + 'pred_of/', max_num)
plotFrames(ofx_pred_test, image_shape, path_test + 'pred_ofx/', max_num)
plotFrames(ofy_pred_test, image_shape, path_test + 'pred_ofy/', max_num)

print '... done'

