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

ofx_train, ofy_train, ofx_test,  ofy_test = loadOpticalFlow()

idx = 100
print 'ploting images ...'
path = predicted_frames_path + 'train/'

preds = numpy.load(preds_path + 'preds_train' + str(idx) + '.npy').flatten()
ofx_train = ofx_train.flatten()
ofy_train = ofy_train.flatten()
print preds.shape
img_preds_numpy = preds #(preds * data_std) + data_mean
# for i in range(img_preds_numpy.shape[0] / frame_dim):
i = 0
pred = preds[i*2*frame_dim:(i+1)*2*frame_dim, None].flatten()
