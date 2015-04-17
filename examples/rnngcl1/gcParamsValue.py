import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

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
            # print x_left[t, :].shape
            # print x_left[[t], :].shape
            vel = self.getVel(x_left[[t], :], x_right[[t], :])
            vels[t, :] = vel

        return vels
