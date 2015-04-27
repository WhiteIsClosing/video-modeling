import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
from collections import OrderedDict

class GraddescentMinibatch(object):
    """ 
    Gradient descent trainer class. 
    """

    def __init__(self, model, data, batch_size, learning_rate, momentum=0.9,
                 normalize_filters=False, rng=None, verbose=True):
        self.model = model
        self.data = data
        self.batch_size = batch_size
        self.learning_rate =\
            theano.shared(numpy.array(learning_rate).astype("float32"))
        self.momentum = momentum 
        self.num_batch = self.data.get_value().shape[0] / batch_size
        self.normalize_filters = normalize_filters 
        self.verbose = verbose

        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        self.index = T.lscalar() 
        self.incs =\
             dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                    dtype=theano.config.floatX), name='inc_'+p.name)) 
                    for p in self.model.params])
        self.inc_updates = OrderedDict()
        self.updates = OrderedDict()
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n

        for param, grad in zip(self.model.params, self.model.grads):
            self.inc_updates[self.incs[param]] =\
                self.momentum * self.incs[param] - self.learning_rate * grad 
            self.updates[param] = param + self.incs[param]

        self.updateincs = theano.function([self.index], self.model.cost, 
                                            updates = self.inc_updates,
                                            givens = {self.model.inputs:\
                                        self.data[self.index*self.batch_size:\
                                            (self.index+1)*self.batch_size]})

        self.trainmodel = theano.function([self.n], self.noop, 
                                            updates = self.updates)
        if verbose:
            print '[GraddescentMinibatch Info]'
            print 'data shape: ' + str(self.data.get_value().shape)
            print 'batch_size: ' + str(self.batch_size)
            print 'num_batch: ' + str(self.num_batch)

    def step(self):
        """
        Optimization round. Return the mean cost. 
        """
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.num_batch-1):
            stepcount += 1.0
            upd = self.updateincs(batch_index)
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*upd
            self.trainmodel(0)
            if self.normalize_filters:
                self.model.normalize_filters()
        return cost

    def reset_incs(self):
        """
        """
        for p in self.model.params:
            self.incs[p].set_value(numpy.zeros(p.get_value().shape, 
                                                dtype=theano.config.floatX))
