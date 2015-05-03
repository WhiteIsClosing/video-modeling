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

wv = model.wv.get_value().T
wxf_left = model.wxf_left.get_value().T
wxf_right = model.wxf_right.get_value().T
bx = model.bx.get_value()
bv = model.bv.get_value()
numpy.save('wm_init', wv)
numpy.save('wfd_left_init', wxf_left)
numpy.save('wfd_right_init', wxf_right)
numpy.save('bd_init', bx)
numpy.save('bm_init', bv)

model.load(models_path + 'model.npy')

toc = clock()
print '...done. time of initializing the model: ' + str(toc - tic)

wv = model.wv.get_value().T
wxf_left = model.wxf_left.get_value().T
wxf_right = model.wxf_right.get_value().T
bx = model.bx.get_value()
bv = model.bv.get_value()
numpy.save('wm', wv)
numpy.save('wfd_left', wxf_left)
numpy.save('wfd_right', wxf_right)
numpy.save('bd', bx)
numpy.save('bm', bv)
