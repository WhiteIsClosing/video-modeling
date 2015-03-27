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
  _seq_len = frame_len * maxframes
  features_numpy = numpy.zeros((seq_num, _seq_len)).astype('float32')
  seq_idx = 0
  for line in list:
    [seqName, _count] = line.split()
    count = int(_count)
    frame_root = data_root + seqName + '/'
    for frm_idx in range(min(count, maxframes)):
      # img = misc.imread(frame_root + seqName + '_' + str(frm_idx) + image_suffix)
      img = misc.imread(frame_root + str(frm_idx) + image_suffix)
      if img.shape != image_shape:
        img = misc.imresize(img, image_shape)
      if img.shape == (image_shape[0], image_shape[1], 3):
        img = img[:, :, 0]
      img_numpy = img.flatten()
      features_numpy[seq_idx, frm_idx*frame_len:(frm_idx+1)*frame_len] = img_numpy
    seq_idx += 1
  return features_numpy


def loadFromImg():
  print 'Loading data...'

  # Read data
  train_list = open(data_root + 'train_list')
  test_list = open(data_root + 'test_list')
  _numtrain = sum(1 for line in train_list)
  _numtest = sum(1 for line in test_list)
  train_list.close()
  test_list.close()
  train_list = open(data_root + 'train_list')
  test_list = open(data_root + 'test_list')

  numtrain = _numtrain * trainframes / numframes
  numtest = _numtest * testframes / numframes


  train_features_numpy = numpy.zeros((numtrain, frame_len * numframes)).astype('float32')
  test_features_numpy = numpy.zeros((numtest, frame_len * numframes)).astype('float32')

  _train_features_numpy = readFrames(train_list, _numtrain, trainframes)
  _test_features_numpy = readFrames(test_list, _numtest, testframes)

  _train_features_numpy = _train_features_numpy.flatten()
  _test_features_numpy = _test_features_numpy.flatten()

  for seq in range(numtrain):
    train_features_numpy[seq, :] = _train_features_numpy[seq*frame_len*numframes:(seq+1)*frame_len*numframes, None].flatten()

  for seq in range(numtest):
    test_features_numpy[seq, :] = _test_features_numpy[seq*frame_len*numframes:(seq+1)*frame_len*numframes, None].flatten()


  # Normalize 
  data_mean = train_features_numpy.mean()
  data_std = train_features_numpy.std()
  train_features_numpy -= data_mean
  train_features_numpy /= data_std 
  # train_features_numpy = train_features_numpy[numpy.random.permutation(numtrain)]
  train_features_theano = theano.shared(train_features_numpy)
  test_features_numpy -= data_mean
  test_features_numpy /= data_std
  test_feature_beginnings = test_features_numpy[:,:frame_len*3]

  print '... done'


  # Save data
  print 'Saving data ...'
  numpy.save(features_root + 'train_features_numpy', train_features_numpy)
  numpy.save(features_root + 'test_features_numpy', test_features_numpy)
  numpy.save(features_root + 'data_mean', data_mean)
  numpy.save(features_root + 'data_std', data_std)
  print '... done'
  return (train_features_numpy, test_features_numpy, numtrain, numtest, data_mean, data_std)


def loadFromFeature():
  train_features_numpy = numpy.load(features_root + 'train_features_numpy.npy')
  test_features_numpy = numpy.load(features_root + 'test_features_numpy.npy')
  data_mean = numpy.load(features_root + 'data_mean.npy')
  data_std = numpy.load(features_root + 'data_std.npy')
  numtrain = train_features_numpy.shape[0]
  numtest = test_features_numpy.shape[0]
  return (train_features_numpy, test_features_numpy, numtrain, numtest, data_mean, data_std)


def loadOpticalFlow():
  numtrain = trainframes / numframes  # TODO
  numtest = testframes / numframes

  train_ofx = numpy.load(data_root + '1/' + 'ofx.npy')[:trainframes, :] # 1 for train, 2 for test
  train_ofy = numpy.load(data_root + '1/' + 'ofy.npy')[:trainframes, :]
  test_ofx = numpy.load(data_root + '2/' + 'ofx.npy')[:testframes, :]
  test_ofy = numpy.load(data_root + '2/' + 'ofy.npy')[:testframes, :]

  train_ofx = numpy.reshape(train_ofx, (numtrain, frame_len * numframes)).astype('float32')
  train_ofy = numpy.reshape(train_ofy, (numtrain, frame_len * numframes)).astype('float32')
  test_ofx = numpy.reshape(test_ofx, (numtest, frame_len * numframes)).astype('float32')
  test_ofy = numpy.reshape(test_ofy, (numtest, frame_len * numframes)).astype('float32')

  # Normalize
  ofx_mean = train_ofx.mean()
  ofx_std = train_ofx.std()
  train_ofx -= ofx_mean
  train_ofx /= ofx_std
  test_ofx -= ofx_mean
  test_ofx /= ofx_std

  ofy_mean = train_ofy.mean()
  ofy_std = train_ofy.std()
  train_ofy -= ofy_mean
  train_ofy /= ofy_std
  test_ofy -= ofy_mean
  test_ofy /= ofy_std

  # train_ofx = numpy.zeros((numtrain, frame_len * numframes)).astype('float32')
  # train_ofy = numpy.zeros((numtrain, frame_len * numframes)).astype('float32')
  # test_ofx = numpy.zeros((numtest, frame_len * numframes)).astype('float32')
  # test_ofy = numpy.zeros((numtest, frame_len * numframes)).astype('float32')

  # _train_ofx = numpy.load(data_root + '1/' + 'ofx.npy')[:trainframes, :] # 1 for train, 2 for test
  # _train_ofy = numpy.load(data_root + '1/' + 'ofy.npy')[:trainframes, :]
  # _test_ofx = numpy.load(data_root + '2/' + 'ofx.npy')[:testframes, :]
  # _test_ofy = numpy.load(data_root + '2/' + 'ofy.npy')[:testframes, :]

  # train_ofx = train_ofx[:trainframes, :]
  # train_ofy = train_ofx[:trainframes, :]
  # test_ofx = test_ofx[:testframes, :]
  # test_ofy = test_ofx[:testframes, :]
  return (train_ofx, train_ofy, test_ofx,  test_ofy)
