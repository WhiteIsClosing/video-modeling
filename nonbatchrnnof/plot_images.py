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


idx = 5
max_num = 100

ofx_train, ofy_train, ofx_test,  ofy_test = loadOpticalFlow()
print 'ploting images ...'
path = predicted_frames_path + 'train/'

preds = numpy.load(preds_path + 'preds_train' + str(idx) + '.npy').flatten()
ofx_train = ofx_train.flatten()
ofy_train = ofy_train.flatten()
print preds.shape
img_preds_numpy = preds #(preds * data_std) + data_mean
for i in range(min(numframes_train, max_num)):
  pred = preds[i*2*frame_dim:(i+1)*2*frame_dim, None].flatten()

  ofx_img = numpy.reshape(pred[0:frame_dim, None], image_shape)
  ofy_img = numpy.reshape(pred[frame_dim:2*frame_dim, None], image_shape)
  trueofx_img = numpy.reshape(ofx_train[i*frame_dim:(i+1)*frame_dim, None], image_shape)
  trueofy_img = numpy.reshape(ofy_train[i*frame_dim:(i+1)*frame_dim, None], image_shape)
  misc.imsave(path + 'ofx/' + str(i) + '.jpeg', ofx_img)
  misc.imsave(path + 'ofy/' + str(i) + '.jpeg', ofy_img)
  misc.imsave(path + 'true_ofx/' + str(i) + '.jpeg', trueofx_img)
  misc.imsave(path + 'true_ofy/' + str(i) + '.jpeg', trueofy_img)

print '... done'

