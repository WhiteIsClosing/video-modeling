import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy
import numpy.random
numpy_rng  = numpy.random.RandomState(1)
from scipy import misc

from hyperParams import *
sys.path.insert(0, project_path)
from utils.load import *
from utils.plot import *
from utils.log import *

print 'loading data ...'

# LOAD DATA
features_train_numpy = \
  loadFrames(data_path + 'train/', image_shape, numframes_train, seq_len)
ofx_train, ofy_train = \
  loadOFs(data_path + 'train/', image_shape, numframes_train, seq_len)
labels_train = numpy.concatenate((ofx_train, ofy_train), axis = 1)

features_test_numpy = \
  loadFrames(data_path + 'test/', image_shape, numframes_test, seq_len)
ofx_test, ofy_test = \
  loadOFs(data_path + 'test/', image_shape, numframes_test, seq_len)
labels_test = numpy.concatenate((ofx_test, ofy_test), axis = 1)


# LOAD PREDICTION
preds_train = numpy.load(pred_path + 'preds_train.npy')
ofx_pred_train = preds_train[:, :frame_dim]
ofy_pred_train = preds_train[:, frame_dim:]
frames_train = features_train_numpy

preds_test = numpy.load(pred_path + 'preds_test.npy')
ofx_pred_test = preds_test[:, :frame_dim]
ofy_pred_test = preds_test[:, frame_dim:]
frames_test = features_test_numpy

print '... done'


#
print 'ploting images ...'

path_train = vis_path + 'train/'
plotFrames(frames_train, image_shape, path_train + 'true_frames/', max_plot)
plotOFs(ofx_train, ofy_train, th_of, 0., image_shape, \
            path_train + 'true_of/', max_plot)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_plot)
plotFrames(ofx_train, image_shape, path_train + 'true_ofx/', max_plot)
plotOFs(ofx_pred_train, ofy_pred_train, th_of, 0., image_shape, \
            path_train + 'pred_of/', max_plot)
plotFrames(ofx_pred_train, image_shape, path_train + 'pred_ofx/', max_plot)
plotFrames(ofy_pred_train, image_shape, path_train + 'pred_ofy/', max_plot)


path_test = vis_path + 'test/'
plotFrames(frames_test, image_shape, path_test + 'true_frames/', max_plot)
plotOFs(ofx_test, ofy_test, th_of, 0., image_shape, \
            path_test + 'true_of/', max_plot)
plotFrames(ofx_test, image_shape, path_test + 'true_ofx/', max_plot)
plotFrames(ofx_test, image_shape, path_test + 'true_ofx/', max_plot)
plotOFs(ofx_pred_test, ofy_pred_test, th_of, 0., image_shape, \
            path_test + 'pred_of/', max_plot)
plotFrames(ofx_pred_test, image_shape, path_test + 'pred_ofx/', max_plot)
plotFrames(ofy_pred_test, image_shape, path_test + 'pred_ofy/', max_plot)

print '... done'

