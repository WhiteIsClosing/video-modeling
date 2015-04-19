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


class GCParamsValue(object):
    def __init__(self, gcParams):
        self.wxf_left = gcParams.wxf_left.get_value()
        self.wxf_right = gcParams.wxf_right.get_value()
        self.wv = gcParams.wv.get_value() 
        self.wvf_left = gcParams.wvf_left.get_value()
        self.wvf_right = gcParams.wvf_right.get_value()
        self.wa = gcParams.wa.get_value()
        self.waf_left = gcParams.waf_left.get_value()
        self.waf_right = gcParams.waf_right.get_value()
        self.wj = gcParams.wj.get_value()
        self.bx = gcParams.bx.get_value()
        self.bv = gcParams.bv.get_value()
        self.ba = gcParams.ba.get_value()
        self.bj = gcParams.bj.get_value()
        self.autonomy = gcParams.autonomy.get_value()

        # grammar-cell velocity mapping unit
        self.frame_left = T.matrix(name='frame_left')
        self.frame_right = T.matrix(name='frame_right')
        self.factor_left = T.dot(self.frame_left, self.wxf_left)
        self.factor_right = T.dot(self.frame_right, self.wxf_right)
        self.vel_ = T.nnet.sigmoid(T.dot(self.factor_left*self.factor_right, self.wv)+self.bv)
        self.getVel = theano.function([self.frame_left, self.frame_right], self.vel_)

    def getVels(self, x):
        numvel = self.wv.shape[1]

        x_left = numpy.concatenate((numpy.zeros((1, x.shape[1])).astype("float32"), x[:-1, :]), axis=0)
        x_right = x
        vels = numpy.zeros((x.shape[0], numvel)).astype("float32")
        for t in range(x.shape[0]):
            vel = self.getVel(x_left[[t], :], x_right[[t], :])
            vels[t, :] = vel

        return vels
