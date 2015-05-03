from patch_gen import *
from scipy import io as sio
from scipy import misc

import sys
project_path  = '/deep/u/kuanfang/video-modeling/'
sys.path.insert(0, project_path)

from utils.plot import *

# hyper paramters
num_seq = 5
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
image = misc.lena()

# generate translation patch sequences
print 'generate translation patch sequences ...'
seqs_trans = translate_gen(image, num_seq, seq_len, patch_size)
print seqs_trans.shape
plotFrames(seqs_trans, (patch_size, patch_size), out_path+'translate/')
print '... done'

# generate rotation patch sequences
print 'generate rotation patch sequences ...'
seqs_rot = rotate_gen(image, num_seq, seq_len, patch_size)
print seqs_rot.shape
plotFramesNoScale(seqs_rot, (patch_size, patch_size), out_path+'rotate/')
print '... done'

# generate scaling patch sequences
print 'generate scaling patch sequences ...'
seqs_scl = scale_gen(image, num_seq, seq_len, patch_size)
print seqs_scl.shape
plotFramesNoScale(seqs_scl, (patch_size, patch_size), out_path+'scale/')
print '... done'
