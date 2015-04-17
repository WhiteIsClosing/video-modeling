# grammar-cell paramters
import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Pgp3layerParamValues(object): 
    def __init__(pgp3layerParams):
        self.wxf_left = pgp3layerParams.wxf_left.get_value()
        self.wxf_right = pgp3layerParams.wxf_right.get_value()
        self.wv = pgp3layerParams.wv.get_value()
        self.wvf_left = pgp3layerParams.wvf_left.get_value()
        self.wvf_right = pgp3layerParams.wvf_right.get_value()
        self.wa = pgp3layerParams.wa.get_value()
        self.waf_left = pgp3layerParams.waf_left.get_value()
        self.waf_right = pgp3layerParams.waf_right.get_value()
        self.wj = pgp3layerParams.wj.get_value()
        self.bx = pgp3layerParams.bx.get_value()
        self.bv = pgp3layerParams.bv.get_value()
        self.ba = pgp3layerParams.ba.get_value()
        self.bj = pgp3layerParams.bj.get_value()
        # self.params = [self.wxf_left, self.wxf_right, self.wv, self.wvf_left, self.wvf_right, self.wa, self.waf_left, self.waf_right, self.wj, self.bx, self.bv, self.ba, self.bj, self.autonomy]
