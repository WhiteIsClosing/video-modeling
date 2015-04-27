import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from params import Params

class GatedAutoencoderAsym(Params):
    """
    Asymmetric Gated Autoencoder
    """
    def __init__(self, 
                    dimdat, dimfac, dimmap,
                    wfd_left=None, wfd_right=None, wmf=None,
                    bd_left=None, bd_right=None, bm=None,
                    output_type='real', corrupt_type='none', corrupt_level=0.0, 
                    numpy_rng=None, theano_rng=None,
                    name=''):
        """
        name : string type name of the model
        # mode : if 'reconstruct' then train for two-way reconstruction
        #         if 'up' then infer mapping unit using two input data
        #         # if 'left' then predict left using right and mapping unit
        #         if 'right' then predict right using left and mapping unit
        """
        self.name = name

        if not numpy_rng:  
            self.numpy_rng = numpy.random.RandomState(1) 
        else:
            self.numpy_rng = numpy_rng
        if not theano_rng:  
            self.theano_rng = RandomStreams(1)

        # hyper parameters
        ########################################################################
        """
        dimdat : Dimension of the data
        dimfac : Dimension of the factors
        dimmap : Dimension of the mapping units
        """
        self.dimdat = dimdat
        self.dimfac = dimfac
        self.dimmap = dimmap

        self.output_type = output_type
        self.corrupt_type = corrupt_type
        self.corrupt_level = corrupt_level

        # trainable parameters
        ########################################################################
        """
        wfd_left :
        wfd_right :
        wmf :
        bd :
        bm :
        """
        # #
        # if wfd_left == None:
        #     self.wfd_left = self.init_param(size=(dimfac, dimdat), scale=.001,  
        #                                 mode='n', name=self.name+':wfd_left')
        # else:
        #     self.wfd_left = wfd_left
        # #
        # if wfd_right == None:
        #     self.wfd_right = self.init_param(size=(dimfac, dimdat), scale=.001, 
        #                                 mode='n', name=self.name+':wfd_right')
        # else:
        #     self.wfd_right = wfd_right
        # #
        # if wmf == None:
        #     self.wmf = self.init_param(size=(dimmap, dimfac), scale=[-3., -2.],
        #                                 mode='u', name=self.name+':wmf')
        # else:
        #     self.wmf = wmf
        # #
        # if bd_left == None:
        #     self.bd_left = self.init_param(size=(dimdat), scale=0.,  
        #                                 mode='r', name=self.name+':bd_left')
        # else:
        #     self.bd_left = bd_left
        # #
        # if bd_right == None:
        #     self.bd_right = self.init_param(size=(dimdat), scale=0.,  
        #                                 mode='r', name=self.name+':bd_right')
        # else:
        #     self.bd_right = bd_right
        # #
        # if bm == None:
        #     self.bm = self.init_param(size=(dimmap), scale=0.,  
        #                                 mode='r', name=self.name+':bm')
        # else:
        #     self.bm = bm

        # self.params = [self.wfd_left, self.wfd_right, self.wmf, self.bd_left, 
        #                 self.bd_right, self.bm]

        wxf_init = numpy_rng.normal(size=(dimfac, dimdat)).astype(theano.config.floatX)*0.001
        wyf_init = numpy_rng.normal(size=(dimfac, dimdat)).astype(theano.config.floatX)*0.001

        self.whf_init = numpy.exp(numpy_rng.uniform(low=-3.0, high=-2.0, size=(dimmap, dimfac)).astype(theano.config.floatX))
        # self.whf_in_init = numpy_rng.uniform(low=-0.01, high=+0.01, size=(nummap, numfac)).astype(theano.config.floatX)
        self.wmf = theano.shared(value = self.whf_init, name='whf')
        # self.whf_in = self.whf #theano.shared(value = self.whf_in_init, name='whf_in')
        self.wfd_left = theano.shared(value = wxf_init, name = 'wxf')
        self.bd_left = theano.shared(value = numpy.zeros(dimdat, dtype=theano.config.floatX), name='bvisX')
        self.wfd_right = theano.shared(value = wyf_init, name = 'wyf')
        self.bd_right = theano.shared(value = numpy.zeros(dimdat, dtype=theano.config.floatX), name='bvisY')
        self.bm = theano.shared(value = 0.0*numpy.ones(dimmap, dtype=theano.config.floatX), name='bmap')
        self.params = [self.wfd_left, self.wfd_right, self.wmf, self.bm, 
                        self.bd_left, self.bd_right]
        # self.params = [self.wxf, self.wyf, self.whf, self.bmap, self.bvisX, self.bvisY]
        # layers 
        ########################################################################
        """
        dat_left : 
        dat_right :
        fac_left :
        fac_right :
        map ::
        """
        
        self.inputs = T.matrix(name=self.name+':inputs') 
        inputs_left = self.inputs[:, :dimdat] 
        inputs_right = self.inputs[:, dimdat:] 
        dat_left = self.inputs[:, :dimdat] 
        dat_right = self.inputs[:, dimdat:] 

        # dat_left = self.corrupt(dat_left, 
        #             self.corrupt_type, self.corrupt_level)
        # dat_right = self.corrupt(dat_right, 
        #             self.corrupt_type, self.corrupt_level)
            
        fac_left = T.dot(dat_left, self.wfd_left.T)
        fac_right = T.dot(dat_right, self.wfd_right.T)
        # map = self.fac_infer(fac_left, fac_right)
        map = T.nnet.sigmoid(T.dot(fac_left * fac_right, self.wmf.T)+self.bm)
        fac_map = T.dot(map, self.wmf)
        # recons_left = self.fac_predict(fac_right, fac_map, 'l')
        # recons_right = self.fac_predict(fac_left, fac_map, 'r')
        recons_left = T.dot(fac_left * fac_map, self.wfd_left) + self.bd_left
        recons_right = T.dot(fac_left * fac_map, self.wfd_right) + self.bd_right
        recons = T.concatenate((recons_left, recons_right), axis=1)

        # cost = T.mean((recons_left - self.inputs[:, :dimdat])**2 +\
        #                     (recons_right - self.inputs[:, dimdat:])**2)
        costpercase = T.sum(0.5*((inputs_left-recons_left)**2)
                                 +0.5*((inputs_right-recons_right)**2), axis=1)
        cost = T.mean(costpercase) 


        grads = T.grad(cost, self.params) 
        self._cost = cost 
        self._grads = grads 
        # functions
        self.f_map = theano.function([self.inputs], map)
        self.f_recons = theano.function([self.inputs], recons)
        self.f_cost = theano.function([self.inputs], cost)
        self.f_grads = theano.function([self.inputs], grads)

    def corrupt(self, raw, corrupt_type, corrupt_level):
        if corrupt_type == 'none' or corrupt_type == None:
            corrupted = raw
        elif corrupt_type == 'zeromask':
            corrupted = self.theano_rng.binomial(size=raw.shape, 
                n=1, p=1.0-corrupt_level, 
                dtype=theano.config.floatX) * raw
        elif corrupt_type == 'mixedmask':
            corrupted = self.theano_rng.binomial(size=raw.shape, 
                n=1, p=1.0-corrupt_level/2, 
                dtype=theano.config.floatX) * raw
            corrupted = (1-self.theano_rng.binomial(size=corrupted.shape, 
                n=1, p=1.0-corrupt_level/2, 
                dtype=theano.config.floatX)) * corrupted
        elif corrupt_type == 'gaussian':
            corrupted = self.theano_rng.normal(size=raw.shape, avg=0.0, 
            std=corrupt_level, dtype=theano.config.floatX) + raw
        else:
            assert False, "corrupt type not understood"
        return corrupted

    def fac_infer(self, fac_left, fac_right):
        """
        Infer the mapping unit given the left and right factors. 
        """
        map = self._fac_infer(fac_left, fac_right, self.wmf, self.bm)
        return map

    def fac_predict(self, fac_in, fac_map, dir='r'):
        """
        Predict one of the data given the factor of the other data and the 
        mapping unit.

        Parameters
        ----------
        dir: str
            Direction of the prediction, 'l' for left and 'r' for right.
        """
        if dir == 'l':
            wfd_out = self.wfd_left
            bd = self.bd_left
        else:
            wfd_out = self.wfd_right
            bd = self.bd_right
        dat_out = self._fac_predict(fac_in, fac_map, wfd_out, bd)
        return dat_out

    def infer(self, dat_left, dat_right):
        """
        Infer the mapping unit given the left and right data. 
        """
        map = self._infer(dat_left, dat_right, 
                            self.wfd_left, self. wfd_right, self.wmf, self.bm)
        return map

    def predict(self, dat_in, map, dir='r'):
        """
        Predict one of the data given the another data and the mapping unit.

        Parameters
        ----------
        dir: str
            Direction of the prediction, 'l' for left and 'r' for right.
        """
        if dir == 'l':
            wfd_in = self.wfd_right
            wfd_out = self.wfd_left
            bd = self.bd_left
        else:
            wfd_in = self.wfd_left
            wfd_out = self.wfd_right
            bd = self.bd_right
        dat_out = self._fac_predict(dat_in, map, 
                                    wfd_in, wfd_out, self.wmf, bd)
        return dat_out

    def _fac_infer(self, fac_left, fac_right, wmf, bm):
        "Called by self.fac_infer()."
        premap = T.dot(fac_left * fac_right, wmf.T) + bm
        map = T.nnet.sigmoid(premap)
        return map

    def _fac_predict(self, fac_in, fac_map, wfd_out, bd):
        "Called by self.predict()."
        dat_out = T.dot(fac_in * fac_map, wfd_out) + bd
        return dat_out

    def _infer(self, dat_left, dat_right, wfd_left, wfd_right, wmf, bm):
        "Called by self.infer()."
        fac_left = T.dot(dat_left, wfd_left.T)
        fac_right = T.dot(dat_right, wfd_right.T)
        # premap = T.dot(fac_left * fac_right, wmf.T) + bm
        # map = T.nnet.sigmoid(premap)
        map = self._fac_infer(fac_left, fac_right, wmf, bm)
        return map

    def _predict(self, dat_in, map, wfd_in, wfd_out, wmf, bd):
        "Called by self.predict()."
        fac_in = T.dot(dat_in, wfd_in.T)
        fac_map = T.dot(map, wmf)
        # dat_out = T.dot(fac_in * fac_map, wfd_out) + bd
        dat_out = self._fac_predict(fac_in, fac_map, wfd_out, bd)
        return dat_out

    def normalize_filters(self):
        """
        Normalize filters. 
        """
        raise Exception('Not impleted yet. ')



