import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

################################################################################

class FactoredGatedAutoencoder(object):
  '''
  Factorer Gated Autoencoder
  '''
  def __init__(self, dimdat, dimfac, dimmap,\
                wfd_left=None, wfd_right=None, wmf=None, autonomy=None,\
                bd=None, bm=None,\
                numpy_rng=None, theano_rng=None,
                name='', mode='up', role='unit'
              ):
    '''
    name :: string type name of the model
    mode :: if 'up' then infer mapping unit using two input data
            if 'right' then predict right data using left data and mapping unit
            if 'left' then predict left data using right data and mapping unit
    role :: if 'unit' then just compute and output results
            if 'independent' then compute cost and grads(gradients)
    '''
    self.name = name
    self.mode = mode
    self.role = role

    # hyper parameters
    ############################################################################
    '''
    dimdat ::  dimension of the data
    dimfac ::  dimension of the factors
    dimmap :: dimension of the mapping units
    '''
    self.dimdat = dimdat
    self.dimfac = dimfac
    self.dimmap = dimmap

    # parameters
    ############################################################################
    '''
    wfd_left :: 
    wfd_right ::
    wmf ::
    bd ::
    bm ::
    autonomy :: 
    ''' 
     if not numpy_rng:  self.numpy_rng = numpy.random.RandomState(1) else:
        self.numpy_rng = numpy_rng
    if not theano_rng:  
        theano_rng = RandomStreams(1)
    #
    if wfd_left == None:
      self.wfd_left = init_weight(self.name+':wfd_left', (dimfac, dimdat))
    else:
      self.wfd_left = wfd_left
    #
    if wfd_right == None:
      self.wfd_right = init_weight(self.name+':wfd_right', (dimfac, dimdat))
    else:
      self.wfd_right = wfd_right
    #
    if wmf == None:
      self.wmf = init_weight(self.name+':wmf', (dimmap, dimfac))
    else:
      self.wmf = wmf
    #
    if bd == None:
      self.bd = init_bias(self.name+':bd', (dimdat)) 
    else:
      self.bd = bd
    #
    if bm == None:
      self.bm = init_bias(self.name+':bm', (dimmap)) 
    else:
      self.bm = bm
    #
    if autonomy == None:
      self.autonomy = theano.shared(value=numpy.array([0.5]).astype("float32"),\
                                     name='autonomy')
    else:
      self.autonomy = autonomy

    self.params = [self.wfd_left, self.wfd_right, self.wmf, self.bd, self.bm,\
                    self.autonomy]

    # layers 
    ############################################################################
    '''
    dat_left :: 
    dat_right ::
    fac_left ::
    fac_right ::
    map ::
    '''
    if self.mode == 'up':
      self.dat_left = T.matrix(name=self.name+':dat_left') 
      self.dat_right = T.matrix(name=self.name+':dat_right') 
      self.fac_left = T.dot(self.dat_left, self.wfd_left.T)
      self.fac_right = T.dot(self.dat_right, self.wfd_right.T)
      self.premap = T.dot(self.fac_left * self.fac_right, self.wmf.T)
      self._map = T.nnet.sigmoid(self.premap)
      self.map = theano.function([self.dat_left, self.dat_right], self._map)

      if self.role == 'independent':
        self.true_map = T.matrix(name=self.name+':true_map')  
        self._cost = T.mean((self._map - self.true_map)**2)
        self._grads = T.grad(self._cost, self.params) # TODO: too many params?
        self.cost =\
          theano.function([self.dat_left, self.dat_right, self.true_map],\
                           self._cost)
        self.grads =\
          theano.function([self.dat_left, self.dat_right, self.true_map],\
                           self._grads)

    elif self.mode == 'right':
      self.dat_left = T.matrix(name=self.name+':dat_left') 
      self.map = T.matrix(name=self.name+':map')  
      self.fac_left = T.dot(self.dat_left, self.wfd_left.T)
      self.fac_map = T.dot(self.map, self.wmf)
      self._dat_right = T.dot(self.fac_left * self.fac_map, self.wfd_right)
      self.dat_right =\
        theano.function([self.dat_left, self.map], self._dat_right)

      if self.role == 'independent':
        self.true_dat_right = T.matrix(name=self.name+':true_dat_right')
        self._cost = T.mean((self._dat_right - self.true_dat_right)**2)
        self._grads = T.grad(self._cost, self.params) # TODO: too many params?
        self.cost =\
          theano.function([self.dat_left, self.dat_right, self.true_map],\
                           self._cost)
        self.grads =\
          theano.function([self.dat_left, self.dat_right, self.true_map],\
                           self._grads)
    else:
      raise Exception('\'' + str(mode) + '\' is not a premitted mode')

  def init_weight(name, size, val=.01):
    '''
    Utility function to initialize theano shared weights
    '''
    return theano.shared(value = val*self.numpy_rng.normal(size=size)\
                          .astype(theano.config.floatX), name=name)

  def init_bias(name, size, val=0.):
    '''
    Utility function to initialize theano shared bias
    '''
    return theano.shared(value = val*numpy.ones(size,\
                          dtype=theano.config.floatX), name=name) 

  def update_params(self, new_params):

    def inplace_update(x, new):
      x[...] = new
      return x

    paramscounter = 0
    for p in self.params:
      pshape = p.get_value().shape
      pnum = numpy.prod(pshape)
      p.set_value(inplace_update(p.get_value(borrow=True), \
            new_params[paramscounter:paramscounter+pnum].reshape(*pshape)), \
            borrow=True)
      paramscounter += pnum 
    return

  def get_params(self):
    return numpy.concatenate([p.get_value(borrow=False).flatten() \
                              for p in self.params])

  def save(self, filename):
    numpy.save(filename, self.get_params())

  def load(self, filename):
    self.update_params(numpy.load(filename))

  def normalize_filters(self):
    raise Exception('Not impleted yet. ')


