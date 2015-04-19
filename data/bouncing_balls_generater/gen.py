import matplotlib
matplotlib.use('Agg') # This line must be put before any matplotlib imports 
import matplotlib.pyplot as plt

import numpy
from numpy import array
from scipy import misc
from std import *
from seq import *
from seq.data_dynamic2 import *

seed(42) # 42 for training-set 
print 'Start to gnerating training data ...'
# n   # number of balls
for n in range(1, 4):
    r=array([1.2]*n)  # radius of the balls
    res = 16  # resolution
    T = 50000  # number of frames

    [frames, ofx, ofy] = bounce_vec_opticalflow(res, n, T, r)

    root = '../bouncing_balls/' + str(n) + 'balls/train/' # 1 for train
    for t in range(T):
        misc.imsave(root + str(t) + '.jpeg', frames[t].reshape(res, res))
        # misc.imsave(root + 'ofx' + str(t) + '.jpeg', ofx[t].reshape(res, res))
        # misc.imsave(root + 'ofy' + str(t) + '.jpeg', ofy[t].reshape(res, res))

    numpy.save(root + 'frames', frames)
    numpy.save(root + 'ofx', ofx)
    numpy.save(root + 'ofy', ofy)
    print 'Finished storing data of ' + str(n) + ' balls. \n'

seed(43) # 43 for test-set
print 'Start to gnerating test data ...'
for n in range(1, 4):
    r=array([1.2]*n)  # radius of the balls
    res = 16  # resolution
    T = 10000  # number of frames

    [frames, ofx, ofy] = bounce_vec_opticalflow(res, n, T, r)

    root = '../bouncing_balls/' + str(n) + 'balls/test/' # 2 for test
    for t in range(T):
        misc.imsave(root + str(t) + '.jpeg', frames[t].reshape(res, res))
        # misc.imsave(root + 'ofx' + str(t) + '.jpeg', ofx[t].reshape(res, res))
        # misc.imsave(root + 'ofy' + str(t) + '.jpeg', ofy[t].reshape(res, res))

    numpy.save(root + 'frames', frames)
    numpy.save(root + 'ofx', ofx)
    numpy.save(root + 'ofy', ofy)
    print 'Finished storing data of ' + str(n) + ' balls. \n'
