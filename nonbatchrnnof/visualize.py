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

idx = 8000
max_num = 10
preds = numpy.load(pred_path + 'preds_train_' + str(idx) + '.npy')
ofx_pred_train = preds[:, :frame_dim]
ofy_pred_train = preds[:, frame_dim:]

features_train_numpy, features_test_numpy = loadFrames()

ofx_train, ofy_train, ofx_test, ofy_test = loadOpticalFlow()
frames_train = features_train_numpy

print 'ploting images ...'

path_train = vis_path + 'train/'
plotFrames(frames_train, image_shape, path_train + 'true_frames/', max_num)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_num)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_num)
plotFrames(ofx_pred_train, image_shape, path_train + 'ofx/', max_num)
plotFrames(ofy_pred_train, image_shape, path_train + 'ofy/', max_num)

print '... done'

