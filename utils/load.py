import numpy
import numpy.random
numpy_rng  = numpy.random.RandomState(1)
from scipy import misc

# Load trainset and testset individually
def loadFrames(path, image_shape, numframes, seq_len):
  '''
  Load raw frames from .npy files. 
  '''
  frame_dim = image_shape[0] * image_shape[1]   # data dimension of each frame
  seq_dim = frame_dim * seq_len                 # data dimension of each sequence
  numseqs = numframes / seq_len                 # number of sequence
  frames = numpy.load(path + 'frames.npy')[:numframes, :] 
  frames = numpy.reshape(frames, (numseqs, seq_dim)).astype('float32')
  return frames


def loadOFs(path, image_shape, numframes, seq_len):
  '''
  Load optical flow from .npy files. 
  '''
  frame_dim = image_shape[0] * image_shape[1]   # data dimension of each frame
  seq_dim = frame_dim * seq_len                 # data dimension of each sequence
  numseqs = numframes / seq_len                 # number of sequence

  ofx = numpy.load(path + 'ofx.npy')[:numframes, :] 
  ofy = numpy.load(path + 'ofy.npy')[:numframes, :]
  ofx = numpy.reshape(ofx, (numseqs, seq_dim)).astype('float32')
  ofy = numpy.reshape(ofy, (numseqs, seq_dim)).astype('float32')
  return ofx, ofy


# # Load trainset and testset together
# def loadFrames(data_path, image_shape, numframes_train, numframes_test, seq_len):
#   '''
#   Load raw frames from .npy files. 
#   '''
#   frame_dim = image_shape[0] * image_shape[1]   # data dimension of each frame
#   seq_dim = frame_dim * seq_len                 # data dimension of each sequence
#   numseqs_train = numframes_train / seq_len     # number of sequence to train
#   numseqs_test = numframes_test / seq_len       # number of sequence to validate
# 
#   frames_train = numpy.load(data_path + 'train/' + 'frames.npy')[:numframes_train, :] 
#   frames_train = numpy.reshape(frames_train, (numseqs_train, seq_dim)).astype('float32')
# 
#   frames_test = numpy.load(data_path + 'test/' + 'frames.npy')[:numframes_test, :] 
#   frames_test = numpy.reshape(frames_test, (numseqs_test, seq_dim)).astype('float32')
# 
#   return frames_train, frames_test
# 
# 
# def loadOFs(data_path, image_shape, numframes_train, numframes_test, seq_len):
#   '''
#   Load optical flow from .npy files. 
#   '''
#   frame_dim = image_shape[0] * image_shape[1]   # data dimension of each frame
#   seq_dim = frame_dim * seq_len                 # data dimension of each sequence
#   numseqs_train = numframes_train / seq_len     # number of sequence to train
#   numseqs_test = numframes_test / seq_len       # number of sequence to validate
# 
#   ofx_train = numpy.load(data_path + 'train/' + 'ofx.npy')[:numframes_train, :] 
#   ofy_train = numpy.load(data_path + 'train/' + 'ofy.npy')[:numframes_train, :]
#   ofx_test = numpy.load(data_path + 'test/' + 'ofx.npy')[:numframes_test, :]
#   ofy_test = numpy.load(data_path + 'test/' + 'ofy.npy')[:numframes_test, :]
# 
#   ofx_train = numpy.reshape(ofx_train, (numseqs_train, seq_dim)).astype('float32')
#   ofy_train = numpy.reshape(ofy_train, (numseqs_train, seq_dim)).astype('float32')
#   ofx_test = numpy.reshape(ofx_test, (numseqs_test, seq_dim)).astype('float32')
#   ofy_test = numpy.reshape(ofy_test, (numseqs_test, seq_dim)).astype('float32')
# 
#   return ofx_train, ofy_train, ofx_test, ofy_test


def shuffle(A, B, grain = 1, seed = 42, en_shuffle = True):
  '''
  Shuffling the data. 
  '''
  if en_shuffle:
    # print 'ToDo: shuffle function is not completed. '
    assert A.shape[0] == B.shape[0]
    num_rows = A.shape[0]
    assert num_rows % grain == 0

    p = numpy.arange(num_rows).reshape((num_rows/grain, grain))
    p = numpy.random.permutation(p)
    p = p.flatten()
  return A[p, :], B[p, :]
