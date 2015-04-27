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

        Params
        ------
        mode: str 
                'normal' or 'n' for drawing from normal distribution, 
                'uniform' or 'u' for drawing from uniform distribution, 
                'repetitive' or 'r' for repeating same values in each element. 
        """
        if mode == 'normal' or mode == 'n':
            weight = theano.shared(value=scale*self.numpy_rng.normal(\
                        size=size).astype(theano.config.floatX), name=name)
        elif mode == 'uniform' or mode == 'u':
            if numpy.size(scale) == 1:
                low = -scale
                high = scale
            elif numpy.size(scale) == 2:
                low = scale[0]
                high = scale[1]
            weight = theano.shared(value=self.numpy_rng.uniform(size=size,
                low=low, high=high).astype(theano.config.floatX), name=name)
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



