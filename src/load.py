import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
from scipy import misc


print 'loading data...'
# User configurations
data_root = '../data/bouncing_balls/1balls/' 
image_suffix = '.jpeg'
image_shape = (16, 16) # single channel images
trainframes = 5000
testframes = 1000
numframes = 10

features_root = '../features/'

# Paramters according to user configurations
frame_len = image_shape[0] * image_shape[1] # single channel images
seq_len = frame_len * numframes

train_list = open(data_root + 'train_list')
test_list = open(data_root + 'test_list')
numtrain = sum(1 for line in train_list)
numtest = sum(1 for line in test_list)
train_list.close()
test_list.close()
train_list = open(data_root + 'train_list')
test_list = open(data_root + 'test_list')
  

# Read frames
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

_numtrain = numtrain
_numtest = numtest
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

# Normalize the data
data_mean = train_features_numpy.mean()
train_features_numpy -= data_mean
data_std = train_features_numpy.std()
train_features_numpy /= data_std 
# train_features_numpy = train_features_numpy[numpy.random.permutation(numtrain)]
train_features_theano = theano.shared(train_features_numpy)
test_features_numpy -= data_mean
test_features_numpy /= data_std
test_feature_beginnings = test_features_numpy[:,:frame_len*3]
print '... done'

print 'Saving data ...'
numpy.save(features_root + 'train_features_numpy', train_features_numpy)
numpy.save(features_root + 'test_features_numpy', test_features_numpy)
print '... done'


# # paramers of the gated autoencoder
# print 'Setting parameters ...'
# 
# numframes_to_train_ = numframes
# numframes_to_predict_ = numframes

print '... done'
