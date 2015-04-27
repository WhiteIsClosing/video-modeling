import matplotlib
matplotlib.use('Agg') # This line must be put before any matplotlib imports 
import matplotlib.pyplot as plt

from collections import OrderedDict
import numpy
from scipy import misc

from hyperParams import *
from load import loadFromFeature
from optimizer import GraddescentMinibatch

# LOAD DATA
features_train_numpy, features_test_numpy, numseqs_train, numseqs_test, \
data_mean, data_std = loadFromFeature()

idx = 0   # Set this parameter before running
maxPlot = 1000

predictions_path = 'predictions/'
predicted_frames_path = 'predicted_frames/'

# prediction = numpy.load(predictions_path + 'prediction' + str(idx) + '.npy').flatten()
# frames_path = predicted_frames_path
prediction_train = numpy.load(predictions_path + 'train/prediction' + str(idx) + '.npy').flatten()
frames_path_train = predicted_frames_path + 'train/'
prediction_test = numpy.load(predictions_path + 'test/prediction' + str(idx) + '.npy').flatten()
frames_path_test = predicted_frames_path + 'test/'

def plotFrames(prediction, frames_path):
  print 'prediction.shape: '
  print prediction.shape
  img_prediction_numpy = (prediction * data_std) + data_mean

  print 'ploting images ...'
  for i in range(min(img_prediction_numpy.shape[0] / frame_dim, maxPlot)):
    pred_img = numpy.reshape(img_prediction_numpy[i*frame_dim:(i+1)*frame_dim, None], image_shape)
    misc.imsave(frames_path + 'pred_' + str(i) + '.jpeg', pred_img)
  print '... done'

plotFrames(prediction_train, frames_path_train)
plotFrames(prediction_test, frames_path_test)
