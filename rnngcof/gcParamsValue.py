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
        #self.params = gcParams.params
