
import numpy
import time
import sys
import subprocess
import os
import random
import theano

from hyperParams import *
from load import loadFromImg
from load import loadOpticalFlow
from optimizer import GraddescentMinibatch

# import rnn
from rnn import rnn

numpy.random.seed(42)
random.seed(423)
seed = 42

# HYPER PARAMETERS 
lr = 0.63 # learning rate

# LOAD DATA
train_features_numpy, test_features_numpy, numtrain, numtest, \
data_mean, data_std = loadFromImg()
