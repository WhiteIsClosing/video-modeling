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

from gae.grammar_cells_l3 import *

print 'loading data ...'


# LOAD DATA
features_train_numpy = \
  loadFrames(data_path + 'train/', image_shape, numframes_train, seq_len)

features_test_numpy = \
  loadFrames(data_path + 'test/', image_shape, numframes_test, seq_len)

true_frames_train = features_train_numpy
true_frames_test = features_test_numpy

# PREPROCESS
data_mean = features_train_numpy.mean()
data_std = features_train_numpy.std()
features_train_numpy -= data_mean
features_train_numpy /= data_std 
features_test_numpy -= data_mean
features_test_numpy /= data_std
# test_feature_beginnings = features_test_numpy[:,:frame_dim*3]
features_train_theano = theano.shared(features_train_numpy)


# LOAD MODEL
model = GrammarCellsL3(\
                        dimx=dimx, 
                        dimfacx=dimfacx, 
                        dimv=dimv, 
                        dimfacv=dimfacv, 
                        dima=dima, 
                        dimfaca=dimfaca, 
                        dimj=dimj, 
                        seq_len=seq_len, 
                        output_type='real', 
                        corrupt_type="zeromask", 
                        corrupt_level=0.0, 
                        numpy_rng=None, theano_rng=None)

print '... initialization done'

model.load(models_path + 'model.npy')


# PLOT FILTER PAIRS
path_u = vis_path + 'wxf_left/'
wxf_left = model.wfx_left.get_value().flatten()
plotFrames(wxf_left, image_shape, path_u, max_plot)
path_v = vis_path + 'wxf_right/'
wxf_right = model.wfx_right.get_value().flatten()
plotFrames(wxf_right, image_shape, path_v, max_plot)

print '... plotting of filter pairs done'

# PLOT PREDICTION
print 'predicting data ...'

# pred_frames_train = model.f_preds(true_frames_train)
# pred_frames_test = model.f_preds(true_frames_test)
pred_frames_train = model.generate(true_frames_train, 20)
pred_frames_test = model.generate(true_frames_test, 20)

print 'plotting data ...'

path_train = vis_path + 'train/'
plotFrames(true_frames_train, image_shape, path_train + 'true_frames/', max_plot)
plotFrames(pred_frames_train, image_shape, path_train + 'pred_frames/', max_plot)

path_test = vis_path + 'test/'
plotFrames(true_frames_test, image_shape, path_test + 'true_frames/', max_plot)
plotFrames(pred_frames_test, image_shape, path_test + 'pred_frames/', max_plot)

print '... done'



