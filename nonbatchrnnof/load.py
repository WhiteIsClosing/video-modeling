import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
from scipy import misc

from hyperParams import *


# Read frames:
def readFrames(list, seq_num, maxframes):
  _seq_dim = frame_dim * maxframes
  features_numpy = numpy.zeros((seq_num, _seq_dim)).astype('float32')
  seq_idx = 0
  for line in list:
    [seqName, _count] = line.split()
    count = int(_count)
    frame_path = data_path + seqName + '/'
    for frm_idx in range(min(count, maxframes)):
      # img = misc.imread(frame_path + seqName + '_' + str(frm_idx) + image_suffix)
      img = misc.imread(frame_path + str(frm_idx) + image_suffix)
      if img.shape != image_shape:
        img = misc.imresize(img, image_shape)
      if img.shape == (image_shape[0], image_shape[1], 3):
        img = img[:, :, 0]
      img_numpy = img.flatten()
      features_numpy[seq_idx, frm_idx*frame_dim:(frm_idx+1)*frame_dim] = img_numpy
    seq_idx += 1
  return features_numpy


# def loadFromImg():
#   print 'Loading data...'
# 
#   # Read data
#   list_train = open(data_path + 'list_train')
#   list_test = open(data_path + 'list_test')
#   _numseqs_train = sum(1 for line in list_train)
#   _numseqs_test = sum(1 for line in list_test)
#   list_train.close()
#   list_test.close()
#   list_train = open(data_path + 'list_train')
#   list_test = open(data_path + 'list_test')
# 
#   numseqs_train = _numseqs_train * numframes_train / seq_len
#   numseqs_test = _numseqs_test * numframes_test / seq_len
# 
# 
#   features_train_numpy = numpy.zeros((numseqs_train, frame_dim * seq_len)).astype('float32')
#   features_test_numpy = numpy.zeros((numseqs_test, frame_dim * seq_len)).astype('float32')
# 
#   _features_train_numpy = readFrames(list_train, _numseqs_train, numframes_train)
#   _features_test_numpy = readFrames(list_test, _numseqs_test, numframes_test)
# 
#   _features_train_numpy = _features_train_numpy.flatten()
#   _features_test_numpy = _features_test_numpy.flatten()
# 
#   for seq in range(numseqs_train):
#     features_train_numpy[seq, :] = _features_train_numpy[seq*frame_dim*seq_len:(seq+1)*frame_dim*seq_len, None].flatten()
# 
#   for seq in range(numseqs_test):
#     features_test_numpy[seq, :] = _features_test_numpy[seq*frame_dim*seq_len:(seq+1)*frame_dim*seq_len, None].flatten()
# 
# 
# 
#   print '... done'
# 
# 
#   # Save data
#   print 'Saving data ...'
#   numpy.save(features_path + 'features_train_numpy', features_train_numpy)
#   numpy.save(features_path + 'features_test_numpy', features_test_numpy)
#   numpy.save(features_path + 'data_mean', data_mean)
#   numpy.save(features_path + 'data_std', data_std)
#   print '... done'
#   return (features_train_numpy, features_test_numpy, numseqs_train, numseqs_test, data_mean, data_std)
# 
# 
# def loadFromFeature():
#   features_train_numpy = numpy.load(features_path + 'features_train_numpy.npy')
#   features_test_numpy = numpy.load(features_path + 'features_test_numpy.npy')
#   data_mean = numpy.load(features_path + 'data_mean.npy')
#   data_std = numpy.load(features_path + 'data_std.npy')
#   numseqs_train = features_train_numpy.shape[0]
#   numseqs_test = features_test_numpy.shape[0]
#   return (features_train_numpy, features_test_numpy, numseqs_train, numseqs_test, data_mean, data_std)
# 
# def loadFromFeature():
#   features_train_numpy = numpy.load(data_path + )
#   features_test_numpy = numpy.load(features_path + 'features_test_numpy.npy')
#   data_mean = numpy.load(features_path + 'data_mean.npy')
#   data_std = numpy.load(features_path + 'data_std.npy')
#   numseqs_train = features_train_numpy.shape[0]
#   numseqs_test = features_test_numpy.shape[0]
#   return (features_train_numpy, features_test_numpy, numseqs_train, numseqs_test, data_mean, data_std)

def loadFrames():

  frames_train = numpy.load(data_path + 'train/' + 'frames.npy')[:numframes_train, :] 
  frames_train = numpy.reshape(frames_train, (numseqs_train, seq_dim)).astype('float32')

  frames_test = numpy.load(data_path + 'test/' + 'frames.npy')[:numframes_test, :] 
  frames_test = numpy.reshape(frames_test, (numseqs_test, seq_dim)).astype('float32')

  return (frames_train, frames_test)


  return (ofx_train, ofy_train, ofx_test,  ofy_test)

def loadOpticalFlow():

  ofx_train = numpy.load(data_path + 'train/' + 'ofx.npy')[:numframes_train, :] 
  ofy_train = numpy.load(data_path + 'train/' + 'ofy.npy')[:numframes_train, :]
  ofx_test = numpy.load(data_path + 'test/' + 'ofx.npy')[:numframes_test, :]
  ofy_test = numpy.load(data_path + 'test/' + 'ofy.npy')[:numframes_test, :]

  ofx_train = numpy.reshape(ofx_train, (numseqs_train, seq_dim)).astype('float32')
  ofy_train = numpy.reshape(ofy_train, (numseqs_train, seq_dim)).astype('float32')
  ofx_test = numpy.reshape(ofx_test, (numseqs_test, seq_dim)).astype('float32')
  ofy_test = numpy.reshape(ofy_test, (numseqs_test, seq_dim)).astype('float32')


  return (ofx_train, ofy_train, ofx_test,  ofy_test)
