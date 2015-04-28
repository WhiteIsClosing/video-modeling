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

from grammar_cells.pgp3layer import *

print 'loading data ...'


# LOAD DATA
features_train_numpy = \
  loadFrames(data_path + 'train/', image_shape, numframes_train, seq_len)
true_frames_train = features_train_numpy
ofx_train, ofy_train = \
  loadOFs(data_path + 'train/', image_shape, numframes_train, seq_len)
labels_train = numpy.concatenate((ofx_train, ofy_train), axis = 1)

features_test_numpy = \
  loadFrames(data_path + 'test/', image_shape, numframes_test, seq_len)
true_frames_test = features_test_numpy
ofx_test, ofy_test = \
  loadOFs(data_path + 'test/', image_shape, numframes_test, seq_len)
labels_test = numpy.concatenate((ofx_test, ofy_test), axis = 1)



# LOAD PREDICTION
pred_frames_train = numpy.load(pred_path + 'pred_frames_train.npy')

pred_frames_test = numpy.load(pred_path + 'pred_frames_test.npy')

print '... done'


# LOAD MODEL
tic = clock()
print 'loading the model ...'
model = Pgp3layer(numvis=frame_dim,
                  numnote=0,
                  numfac=numfac,
                  numvel=numvel,
                  numvelfac=numvelfac,
                  numacc=numacc,
                  numaccfac=numaccfac,
                  numjerk=numjerk,
                  seq_len_to_train=seq_len_to_train,
                  seq_len_to_predict=seq_len_to_predict,
                  output_type='real',
                  vis_corruption_type='zeromask',
                  vis_corruption_level=0.0,
                  numpy_rng=numpy_rng,
                  theano_rng=theano_rng)

model.load(models_path + 'model.npy')

toc = clock()
print '...done. time of initializing the model: ' + str(toc - tic)


# PLOT FILTER PAIRS
path_u = vis_path + 'wxf_left/'
wxf_left = model.wxf_left.get_value().T.flatten()
plotFrames(wxf_left, image_shape, path_u, max_plot)
path_u = vis_path + 'wxf_right/'
wxf_right = model.wxf_right.get_value().T.flatten()
plotFrames(wxf_right, image_shape, path_u, max_plot)


# PLOT PREDICTION
print 'ploting frames and oftical flows ...'

path_train = vis_path + 'train/'
plotFrames(true_frames_train, image_shape, path_train + 'true_frames/', max_plot)
plotOFs(ofx_train, ofy_train, th_of, 0., image_shape, \
            path_train + 'true_of/', max_plot)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_plot)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_plot)

plotFrames(pred_frames_train, image_shape, path_train + 'pred_frames/', max_plot)

print '... done'



