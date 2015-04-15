import pylab
from collections import OrderedDict
import numpy
import numpy.random
numpy_rng  = numpy.random.RandomState(1)
from scipy import misc

from hyperParams import *
from load import *
from logInfo import *
from plot import *
from gcParams import *
from gcParamsValue import *

gcParams = GCParams(numvis=frame_dim,
                    numnote=0,
                    numfac=numfac_,
                    numvel=numvel_,
                    numvelfac=numvelfac_,
                    numacc=numacc_,
                    numaccfac=numaccfac_,
                    numjerk=numjerk_,
                    seq_len_to_train=seq_len_to_train_,
                    seq_len_to_predict=seq_len_to_predict_,
                    output_type='real',
                    vis_corruption_type='zeromask',
                    vis_corruption_level=0.0,
                    numpy_rng=numpy_rng,
                    theano_rng=theano_rng)
gcParams.load(gc_path + 'model.npy')

gcParamsValue = GCParamsValue(gcParams)

print gcParams.wxf_left.get_value()
print gcParamsValue.wxf_left

