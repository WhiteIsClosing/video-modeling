from patch_gen import *
import numpy
from scipy import io as sio
from scipy import misc

import sys
project_path  = '/deep/u/kuanfang/video-modeling/'
sys.path.insert(0, project_path)

from utils.plot import *

# hyper paramters
num_seq = 5000
seq_len = 10
patch_size = 16 
out_path = '../patch_data/'

# load the IMAGES_CELL.mat
images_cell = sio.loadmat('IMAGES_cell.mat')['IMAGES']
images = [None] * len(images_cell)
i = 0
for cell in images_cell:
    images[i] = cell[0]
    i += 1

# image = images[2]

# generate translation patch sequences
print 'generate translation patch sequences ...'
seqs_trans = numpy.zeros((len(images_cell)*num_seq, seq_len*(patch_size**2)))
for i in range(len(images_cell)):
    seqs_trans[i*num_seq:(i+1)*num_seq]\
        = translate_gen(images[i], num_seq, seq_len, patch_size)
numpy.random.shuffle(seqs_trans)
print seqs_trans.shape
numpy.save(out_path+'translation', seqs_trans)
# plotFrames(seqs_trans, (patch_size, patch_size), out_path+'translate/')
print '... done'

# generate rotation patch sequences
print 'generate rotation patch sequences ...'
seqs_rot = numpy.zeros((len(images_cell)*num_seq, seq_len*(patch_size**2)))
for i in range(len(images_cell)):
    seqs_rot[i*num_seq:(i+1)*num_seq]\
        = rotate_gen(images[i], num_seq, seq_len, patch_size)
numpy.random.shuffle(seqs_rot)
print seqs_rot.shape
numpy.save(out_path+'rotation', seqs_rot)
# plotFramesNoScale(seqs_rot, (patch_size, patch_size), out_path+'rotate/')
print '... done'

# generate scaling patch sequences
print 'generate scaling patch sequences ...'
seqs_scl = numpy.zeros((len(images_cell)*num_seq, seq_len*(patch_size**2)))
for i in range(len(images_cell)):
    seqs_scl[i*num_seq:(i+1)*num_seq]\
        = scale_gen(images[i], num_seq, seq_len, patch_size)
numpy.random.shuffle(seqs_scl)
print seqs_scl.shape
numpy.save(out_path+'scaling', seqs_scl)
# plotFramesNoScale(seqs_scl, (patch_size, patch_size), out_path+'scale/')
print '... done'


# image = misc.lena()

# # generate translation patch sequences
# print 'generate translation patch sequences ...'
# seqs_trans = translate_gen(image, num_seq, seq_len, patch_size)
# print seqs_trans.shape
# plotFrames(seqs_trans, (patch_size, patch_size), out_path+'translate/')
# print '... done'
# 
# # generate rotation patch sequences
# print 'generate rotation patch sequences ...'
# seqs_rot = rotate_gen(image, num_seq, seq_len, patch_size)
# print seqs_rot.shape
# plotFramesNoScale(seqs_rot, (patch_size, patch_size), out_path+'rotate/')
# print '... done'
# 
# # generate scaling patch sequences
# print 'generate scaling patch sequences ...'
# seqs_scl = scale_gen(image, num_seq, seq_len, patch_size)
# print seqs_scl.shape
# plotFramesNoScale(seqs_scl, (patch_size, patch_size), out_path+'scale/')
# print '... done'
