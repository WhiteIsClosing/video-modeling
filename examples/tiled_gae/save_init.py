import numpy
import time
import sys
import random
import theano
from time import clock

from hyper_params import *
sys.path.insert(0, project_path)
from utils.load import *
from utils.log import *

from gae.gated_autoencoder import *
from gae.solver import *

seed = 42
numpy.random.seed(seed)
random.seed(seed)

logInfo = LogInfo('LOG.txt')

# INITIALIZATION
model = GatedAutoencoder(
                            dimdat=dimdat,
                            dimfac=dimfac,
                            dimmap=dimmap,
                            corrupt_type='zeromask', 
                            corrupt_level=0.3, 
                            )

wv = model.wmf.get_value()
wxf_left = model.wfd_left.get_value()
wxf_right = model.wfd_right.get_value()
bx = model.bd.get_value()
bv = model.bm.get_value()
numpy.save('wm_init', wv)
numpy.save('wfd_left_init', wxf_left)
numpy.save('wfd_right_init', wxf_right)
numpy.save('bd_init', bx)
numpy.save('bm_init', bv)
