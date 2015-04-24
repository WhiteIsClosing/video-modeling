import numpy, pylab
import cPickle

import theano
import theano.tensor as T
import theano.tensor.signal.conv 
from theano.tensor.shared_randomstreams import RandomStreams

SMALL = 0.000001

class GrammarCellsL1(object):
    """
    1-layer grammar cells
    """
    def __init__(self, dimvisX, dimvisY, dimfac, dimmap, output_type='real', 
                    corruption_type='zeromask', corruption_level=0.0, 
                    wxf_init=None, wyf_init=None,
                    numpy_rng=None, theano_rng=None):
        if not numpy_rng:  
            numpy_rng = numpy.random.RandomState(1)
        if not theano_rng:  
            theano_rng = RandomStreams(1)

        if wxf_init is None:
            wxf_init = numpy_rng.normal(size=(numvisX, numfac)).astype(theano.config.floatX)*0.001
        if wyf_init is None:
            wyf_init = numpy_rng.normal(size=(numvisY, numfac)).astype(theano.config.floatX)*0.001

        self.whf_init = numpy.exp(numpy_rng.uniform(low=-3.0, high=-2.0, size=(nummap, numfac)).astype(theano.config.floatX))
        self.whf_in_init = numpy_rng.uniform(low=-0.01, high=+0.01, size=(nummap, numfac)).astype(theano.config.floatX)
        self.whf = theano.shared(value = self.whf_init, name='whf')
        self.whf_in = self.whf #theano.shared(value = self.whf_in_init, name='whf_in')
        self.wxf = theano.shared(value = wxf_init, name = 'wxf')
        self.bvisX = theano.shared(value = numpy.zeros(numvisX, dtype=theano.config.floatX), name='bvisX')
        self.wyf = theano.shared(value = wyf_init, name = 'wyf')
        self.bvisY = theano.shared(value = numpy.zeros(numvisY, dtype=theano.config.floatX), name='bvisY')
        self.bmap = theano.shared(value = 0.0*numpy.ones(nummap, dtype=theano.config.floatX), name='bmap')
        self.params = [self.wxf, self.wyf, self.whf, self.bmap, self.bvisX, self.bvisY]

        self.inputsX = self.inputs[:, :numvisX]
        self.inputsY = self.inputs[:, numvisX:]
