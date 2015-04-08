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
from utils import plotFrames

idx = 800
max_num = 100

print 'ploting images ...'

features_train_numpy, features_test_numpy = loadFrames()
ofx_train, ofy_train, ofx_test, ofy_test = loadOpticalFlow()

# train
frames_train = features_train_numpy
preds_train = numpy.load(pred_path + 'preds_train_' + str(idx) + '.npy')
ofx_pred_train = preds_train[:, :frame_dim]
ofy_pred_train = preds_train[:, frame_dim:]

path_train = vis_path + 'train/'
plotFrames(frames_train, image_shape, path_train + 'true_frames/', max_num)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_num)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_num)
plotFrames(ofx_pred_train, image_shape, path_train + 'ofx/', max_num)
plotFrames(ofy_pred_train, image_shape, path_train + 'ofy/', max_num)

# test
frames_test = features_test_numpy
preds_test = numpy.load(pred_path + 'preds_test_' + str(idx) + '.npy')
ofx_pred_test = preds_test[:, :frame_dim]
ofy_pred_test = preds_test[:, frame_dim:]
path_test = vis_path + 'test/'

plotFrames(frames_test, image_shape, path_test + 'true_frames/', max_num)
plotFrames(ofx_test, image_shape, path_test + 'true_ofx/', max_num)
plotFrames(ofx_test, image_shape, path_test + 'true_ofx/', max_num)
plotFrames(ofx_pred_test, image_shape, path_test + 'ofx/', max_num)
plotFrames(ofy_pred_test, image_shape, path_test + 'ofy/', max_num)

print '... done'

