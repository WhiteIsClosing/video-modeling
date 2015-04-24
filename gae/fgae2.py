import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class FGAE(object):
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
            self.wfd_left = self.init_weight('wfd_left', (dimfac, dimdat))
        else:
            self.wfd_left = wfd_left
        #
        if wfd_right == None:
            self.wfd_right = self.init_weight('wfd_right', (dimfac, dimdat))
        else:
            self.wfd_right = wfd_right
        #
        if wmf == None:
            self.wmf = self.init_weight('wmf', (dimmap, dimfac))
        else:
            self.wmf = wmf
        #
        if bd == None:
            self.bd = self.init_bias('bd', (dimdat)) 
        else:
            self.bd = bd
        #
        if bm == None:
            self.bm = self.init_bias('bm', (dimmap)) 
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
            self.premap =\
                T.dot(self.fac_left - self.fac_right, self.wmf.T) + self.bm
            self._map = numpy.ones(dimmap) - T.nnet.sigmoid(self.premap)
            self.fac_map = T.dot(self._map, self.wmf)
            self._recons_left =\
                T.dot(self.fac_right * self.fac_map, self.wfd_left) + self.bd
            self._recons_right =\
                T.dot(self.fac_left * self.fac_map, self.wfd_right) + self.bd
            self._recons =\
                T.concatenate((self._recons_left, self._recons_right), axis=1)
            self._cost = T.mean((self._recons_left - self.dat_left)**2 +\
                                (self._recons_right - self.dat_right)**2)
            self._grads = T.grad(self._cost, self.params) 
            # functions
            self.map = theano.function([self.dat_left, self.dat_right], 
                                        self._map)
            self.recons_left = theano.function([self.dat_left, self.dat_right],
                                                    self._recons_left)
            self.recons_right = theano.function([self.dat_left,self.dat_right],
                                                    self._recons_right)
            # self.map = theano.function([self.self.inputs], self._map)
            # self.recons_left = theano.function([self.inputs], 
            #                                        self._recons_left)
            # self.recons_right = theano.function([self.inputs], 
            #                                        self._recons_rith)
            self.cost = theano.function([self.inputs], self._cost)
            self.grads = theano.function([self.inputs], self._grads)
            self.predict = theano.function([self.inputs], self._recons)

        elif self.mode == 'map':
            self.dat_left = T.matrix(name=self.name+':dat_left') 
            self.dat_right = T.matrix(name=self.name+':dat_right') 
            self.fac_left = T.dot(self.dat_left, self.wfd_left.T)
            self.fac_right = T.dot(self.dat_right, self.wfd_right.T)
            self.premap = T.dot(self.fac_left * self.fac_right, self.wmf.T)
            self._map = T.nnet.sigmoid(self.premap)
            self.map =\
                theano.function([self.dat_left, self.dat_right], self._map)

        elif self.mode == 'predict':
            self.dat_left = T.matrix(name=self.name+':dat_left') 
            self.map = T.matrix(name=self.name+':map')  
            self.fac_left = T.dot(self.dat_left, self.wfd_left.T)
            self.fac_map = T.dot(self.map, self.wmf)
            self._dat_right =\
                T.dot(self.fac_left * self.fac_map, self.wfd_right)
            self.predict =\
                theano.function([self.dat_left, self.map], self._dat_right)

        else:
            raise Exception('\'' + str(mode) + '\' is not a premitted mode')

    def init_weight(self, name, size, val=.01):
        """
        Utility function to initialize theano shared weights.
        """
        return theano.shared(value = val*self.numpy_rng.normal(size=size)\
                      .astype(theano.config.floatX), name=self.name+':'+name)

    def init_bias(self, name, size, val=0.):
        """
        Utility function to initialize theano shared bias.
        """
        return theano.shared(value = val*numpy.ones(size,
                          dtype=theano.config.floatX), name=self.name+':'+name) 

    def set_params(self, new_params):
        """
        Set all values in self.params to new_params.
        """

        def inplace_update(x, new):
            x[...] = new
            return x

        params_counter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplace_update(p.get_value(borrow=True),
                        new_params[params_counter:params_counter+pnum]\
                        .reshape(*pshape)), borrow=True)
            params_counter += pnum 
        return

    def get_params(self):
        """
        Return a concatenation of self.params. 
        """
        return numpy.concatenate([p.get_value(borrow=False).flatten()
                                    for p in self.params])

    def save(self, filename):
        """
        Save self.params.
        """
        numpy.save(filename, self.get_params())

    def load(self, filename):
        """
        Load self.params. 
        """
        self.set_params(numpy.load(filename))

    def normalize_filters(self):
        """
        """
        raise Exception('Not impleted yet. ')


