# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import sys
import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from scipy import misc
from time import clock
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)

from hyper_params import *
sys.path.insert(0, project_path)
from utils.load import *
from utils.plot import *
from utils.log import *

from gae.gated_autoencoder import *

print 'loading data ...'


# LOAD DATA
features_train_numpy = \
  loadFrames(data_path + 'train/', image_shape, numframes_train, seq_len)

features_test_numpy = \
  loadFrames(data_path + 'test/', image_shape, numframes_test, seq_len)


# LOAD PREDICTION
pred_frames_train = numpy.load(pred_path + 'recons.npy')

print '... done'


model = GatedAutoencoder(dimdat=dimdat,
                            dimfac=dimfac,
                            dimmap=dimmap)
model.load(models_path + 'model.npy')

# PLOT FILTER PAIRS
path_u = vis_path + 'wxf_left/'
wfd_left = model.wfd_left.get_value().T.flatten()
plotFrames(wfd_left, image_shape, path_u, max_plot)
path_v = vis_path + 'wxf_right/'
wfd_right = model.wfd_right.get_value().T.flatten()
plotFrames(wfd_right, image_shape, path_v, max_plot)


# PLOT PREDICTION
print 'ploting frames and oftical flows ...'

path_train = vis_path + 'train/'
plotFrames(features_train_numpy, image_shape, path_train + 'true_frames/', max_plot)
plotFrames(pred_frames_train, image_shape, path_train + 'pred_frames/', max_plot)

print '... done'



