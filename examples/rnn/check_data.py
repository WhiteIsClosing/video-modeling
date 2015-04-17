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


path = 'check/'
max_num = 10

features_train_numpy, features_test_numpy = loadFrames()

ofx_train, ofy_train, ofx_test, ofy_test = loadOpticalFlow()
labels_train = numpy.concatenate((ofx_train, ofy_train), axis = 1)
labels_test = numpy.concatenate((ofx_test, ofy_test), axis = 1)

frames_train = features_train_numpy.flatten()
labels_train = labels_train.flatten()

ofx_train = ofx_train.flatten()
ofy_train = ofy_train.flatten()

for i in range(min(numframes_train, max_num)):
  frame = frames_train[i*frame_dim:(i+1)*frame_dim] 
  # pred = labels_train[i*frame_dim:(i+2)*frame_dim]

  # ofx_img = numpy.reshape(pred[0:frame_dim, None], image_shape)
  # ofy_img = numpy.reshape(pred[frame_dim:2*frame_dim, None], image_shape)
  trueofx_img = numpy.reshape(ofx_train[i*frame_dim:(i+1)*frame_dim, None], image_shape)
  trueofy_img = numpy.reshape(ofy_train[i*frame_dim:(i+1)*frame_dim, None], image_shape)
  misc.imsave(path + 'ofx/' + str(i) + '.jpeg', trueofx_img)
  misc.imsave(path + 'ofy/' + str(i) + '.jpeg', trueofy_img)

  frame_img = numpy.reshape(frame, image_shape)
  misc.imsave(path + 'frames/' + str(i) + '.jpeg', frame_img)


print '... done'

