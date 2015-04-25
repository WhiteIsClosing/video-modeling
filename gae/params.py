import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

class Params(object):
    """
    Base class: Params
    """
    def __init__(self):
        """
        Init function. 
        """
        self.params =[]

    def init_param(self, size, scale=.01, mode='n', name=''):
        """
        Utility function to initialize theano shared weights.
        mode: 
            'normal' for drawing from normal distribution, 
            'uniform' for drawing from uniform distribution, 
            'repetitive' for repeating same values in each element. 
        """
        if mode == 'normal' or mode == 'n':
            weight = theano.shared(value = scale*self.numpy_rng.normal(\
                        size=size).astype(theano.config.floatX), name=name)
        elif mode == 'uniform' or mode == 'u':
            weight = theano.shared(value = scale*self.numpy_rng.uniform(\
                        size=size).astype(theano.config.floatX), name=name)
        elif mode == 'repetitive' or mode == 'r':
            weight = theano.shared(value = scale*numpy.ones(size,
                        dtype=theano.config.floatX), name=name) 
        else:
            raise Exception('\''+str(mode)+'\'' + ' is not a valid mode. ')
        return weight  

    def set_params(self, new_params):
        """
        Set all values in self.params to new_params.
        """

        def inplace_set(x, new):
            x[...] = new
            return x

        params_counter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplace_set(p.get_value(borrow=True),
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



