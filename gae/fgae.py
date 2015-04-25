import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from params import Params

class FGAE(Params):
    """
    Factored Gated Autoencoder
    """
    def __init__(self, dimdat, dimfac, dimmap,
                    wfd_left=None, wfd_right=None, wmf=None, autonomy=None,
                    bd=None, bm=None,
                    # output_type='real', corrupt_type=None, corrupt_level=0.0, 
                    numpy_rng=None, theano_rng=None,
                    name='', mode='reconstruct'):
        """
        name :: string type name of the model
        mode :: if 'reconstruct' then train for two-way reconstruction
                if 'up' then infer mapping unit using two input data
                # if 'left' then predict left using right and mapping unit
                if 'right' then predict right using left and mapping unit
        """
        self.name = name
        self.mode = mode

        # hyper parameters
        ########################################################################
        """
        dimdat ::  dimension of the data
        dimfac ::  dimension of the factors
        dimmap :: dimension of the mapping units
        """
        self.dimdat = dimdat
        self.dimfac = dimfac
        self.dimmap = dimmap

        # parameters
        ########################################################################
        """
        wfd_left :: 
        wfd_right ::
        wmf ::
        bd ::
        bm ::
        """
        if not numpy_rng:  
            self.numpy_rng = numpy.random.RandomState(1) 
        else:
            self.numpy_rng = numpy_rng
        if not theano_rng:  
            theano_rng = RandomStreams(1)
        #
        if wfd_left == None:
            self.wfd_left = self.init_param(size=(dimfac, dimdat), scale=.01,  
                                        mode='n', name=self.name+':wfd_left')
        else:
            self.wfd_left = wfd_left
        #
        if wfd_right == None:
            self.wfd_right = self.init_param(size=(dimfac, dimdat), scale=.01,  
                                        mode='n', name=self.name+':wfd_right')
        else:
            self.wfd_right = wfd_right
        #
        if wmf == None:
            self.wmf = self.init_param(size=(dimmap, dimfac), scale=.01,  
                                        mode='n', name=self.name+':wmf')
        else:
            self.wmf = wmf
        #
        if bd == None:
            self.bd = self.init_param(size=(dimdat), scale=0.,  
                                        mode='r', name=self.name+':bd')
        else:
            self.bd = bd
        #
        if bm == None:
            self.bm = self.init_param(size=(dimmap), scale=0.,  
                                        mode='r', name=self.name+':bm')
        else:
            self.bm = bm

        self.params =[self.wfd_left, self.wfd_right, self.wmf, self.bd, self.bm]

        # layers 
        ########################################################################
        """
        dat_left :: 
        dat_right ::
        fac_left ::
        fac_right ::
        map ::
        """
        
        if self.mode == 'reconstruct':
            # data layers
            self.inputs = T.matrix(name=self.name+':inputs') 
            # self.dat_left = T.matrix(name=self.name+':dat_left') 
            # self.dat_right = T.matrix(name=self.name+':dat_right') 
            self.dat_left = self.inputs[:, :dimdat] 
            self.dat_right = self.inputs[:, dimdat:] 
            self.fac_left = T.dot(self.dat_left, self.wfd_left.T)
            self.fac_right = T.dot(self.dat_right, self.wfd_right.T)
            # self.map = self.infer(self.dat_left, self.dat_right, 
            #             self.wfd_left, self.wfd_right, self.wmf, self.bm)
            self.map = self.fac_infer(self.fac_left, self.fac_right, 
                                        self.wmf, self.bm)
            self.fac_map = T.dot(self.map, self.wmf)
            self.recons_left =\
                self.fac_predict(self.fac_right, self.fac_map, 
                                    self.wfd_left, self.bd)
                # self.predict(self.dat_right, self.map, 
                #             self.wfd_right, self.wfd_left, self.wmf, self.bd)
            self.recons_right =\
                self.fac_predict(self.fac_left, self.fac_map, 
                                    self.wfd_right, self.bd)
                # self.predict(self.dat_left, self.map, 
                #             self.wfd_left, self.wfd_right, self.wmf, self.bd)
            self.recons =\
                T.concatenate((self.recons_left, self.recons_right), axis=1)
            self._cost = T.mean((self.recons_left - self.dat_left)**2 +\
                                (self.recons_right - self.dat_right)**2)
            self._grads = T.grad(self._cost, self.params) 
            # functions
            # self.recons_left = theano.function([self.dat_left, self.dat_right]
            #                                         self.recons_left)
            # self.recons_right = theano.function([self.dat_left,self.dat_right]
            #                                         self.recons_right)
            # self.map = theano.function([self.self.inputs], self._map)
            # self.recons_left = theano.function([self.inputs], 
            #                                        self._recons_left)
            # self.recons_right = theano.function([self.inputs], 
            #                                        self._recons_rith)
            self.cost = theano.function([self.inputs], self._cost)
            self.grads = theano.function([self.inputs], self._grads)
            self.predict = theano.function([self.inputs], self.recons)

        # elif self.mode == 'map':
        #     self.dat_left = T.matrix(name=self.name+':dat_left') 
        #     self.dat_right = T.matrix(name=self.name+':dat_right') 
        #     self.fac_left = T.dot(self.dat_left, self.wfd_left.T)
        #     self.fac_right = T.dot(self.dat_right, self.wfd_right.T)
        #     self.premap = T.dot(self.fac_left * self.fac_right, self.wmf.T)
        #     self._map = T.nnet.sigmoid(self.premap)
        #     self.map =\
        #         theano.function([self.dat_left, self.dat_right], self._map)

        # elif self.mode == 'predict':
        #     self.dat_left = T.matrix(name=self.name+':dat_left') 
        #     self.map = T.matrix(name=self.name+':map')  
        #     self.fac_left = T.dot(self.dat_left, self.wfd_left.T)
        #     self.fac_map = T.dot(self.map, self.wmf)
        #     self._dat_right =\
        #         T.dot(self.fac_left * self.fac_map, self.wfd_right)
        #     self.predict =\
        #         theano.function([self.dat_left, self.map], self._dat_right)

        else:
            raise Exception('\'' + str(mode) + '\' is not a premitted mode')

    def fac_infer(self, fac_left, fac_right, wmf, bm):
        """
        Infer the mapping unit given the left and right factors. 
        """
        premap = T.dot(fac_left * fac_right, wmf.T) + bm
        map = T.nnet.sigmoid(premap)
        return map

    def fac_predict(self, fac_in, fac_map, wfd_out, bd):
        """
        Predict one of the data given the factor of the other data and the 
        mapping unit.
        """
        dat_out = T.dot(fac_in * fac_map, wfd_out) + bd
        return dat_out

    def infer(self, dat_left, dat_right, wfd_left, wfd_right, wmf, bm):
        """
        Infer the mapping unit given the left and right data. 
        """
        fac_left = T.dot(dat_left, wfd_left.T)
        fac_right = T.dot(dat_right, wfd_right.T)
        premap = T.dot(fac_left * fac_right, wmf.T) + bm
        map = T.nnet.sigmoid(premap)
        return map

    def predict(self, dat_in, map, wfd_in, wfd_out, wmf, bd):
        """
        Predict one of the data given the another data and the mapping unit.
        """
        fac_in = T.dot(dat_in, wfd_in.T)
        fac_map = T.dot(map, wmf)
        dat_out = T.dot(fac_in * fac_map, wfd_out) + bd
        return dat_out

    def normalize_filters(self):
        """
        Normalize filters. 
        """
        raise Exception('Not impleted yet. ')


