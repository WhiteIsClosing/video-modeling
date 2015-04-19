import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class GCParams(object):
    def __init__(self, numvis, numnote, numfac, numvel, numvelfac, numacc, numaccfac, numjerk):
        self.numvis = numvis
        self.numnote = numnote
        self.numfac = numfac
        self.numvel = numvel
        self.numvelfac = numvelfac
        self.numacc = numacc
        self.numaccfac = numaccfac
        self.numjerk = numjerk

        self.numpy_rng = numpy.random.RandomState(1)
        theano_rng = RandomStreams(1)

        self.wxf_left = theano.shared(value = self.numpy_rng.normal(size=(numvis+numnote, numfac)).astype(theano.config.floatX)*0.01, name='wxf_left')
        self.wxf_right = theano.shared(value = self.numpy_rng.normal(size=(numvis+numnote, numfac)).astype(theano.config.floatX)*0.01, name='wxf_right')
        self.wv = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numfac, numvel)).astype(theano.config.floatX), name='wv')
        self.wvf_left = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvel, numvelfac)).astype(theano.config.floatX), name='wvf_left')
        self.wvf_right = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvel, numvelfac)).astype(theano.config.floatX), name='wvf_right')
        self.wa = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numvelfac, numacc)).astype(theano.config.floatX), name='wa')
        self.waf_left = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numacc, numaccfac)).astype(theano.config.floatX), name='waf_left')
        self.waf_right = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numacc, numaccfac)).astype(theano.config.floatX), name='waf_right')
        self.wj = theano.shared(value = self.numpy_rng.uniform(low=-0.01, high=+0.01, size=(numaccfac, numjerk)).astype(theano.config.floatX), name='wj')
        self.bx = theano.shared(value = 0.0*numpy.ones(numvis+numnote, dtype=theano.config.floatX), name='bx')
        self.bv = theano.shared(value = 0.0*numpy.ones(numvel, dtype=theano.config.floatX), name='bv')
        self.ba = theano.shared(value = 0.0*numpy.ones(numacc, dtype=theano.config.floatX), name='ba')
        self.bj = theano.shared(value = 0.0*numpy.ones(numjerk, dtype=theano.config.floatX), name='bj')
        self.autonomy = theano.shared(value=numpy.array([0.5]).astype("float32"), name='autonomy')
        self.params = [self.wxf_left, self.wxf_right, self.wv, self.wvf_left, self.wvf_right, self.wa, self.waf_left, self.waf_right, self.wj, self.bx, self.bv, self.ba, self.bj, self.autonomy]

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value(borrow=False).flatten() for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))
