import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
from collections import OrderedDict

class GraddescentMinibatch(object):
    """ Gradient descent trainer class. """

    def __init__(self, model, data, batchsize, learningrate, momentum=0.9, normalizefilters=False, rng=None, verbose=True):
        self.model         = model
        self.data          = data
        self.learningrate  = theano.shared(numpy.array(learningrate).astype("float32"))
        self.verbose       = verbose
        self.batchsize     = batchsize
        self.numbatches    = self.data.get_value().shape[0] / batchsize

        # Debug: 
        print '[GraddescentMinibatch Info]'
        print 'self.data.get_value().shape: '
        print self.data.get_value().shape
        print 'batchsize: '
        print batchsize
        print 'self.numbatches: '
        print self.numbatches

        self.momentum      = momentum 
        self.normalizefilters = normalizefilters 
        if rng is None:
            self.rng = numpy.random.RandomState(1)
        else:
            self.rng = rng

        # self.epochcount = 0
        self.index = T.lscalar() 
        self.incs = dict([(p, theano.shared(value=numpy.zeros(p.get_value().shape, 
                            dtype=theano.config.floatX), name='inc_'+p.name)) for p in self.model.params])
        self.inc_updates = OrderedDict() #{} :TODO
        self.updates = OrderedDict() #{}
        self.n = T.scalar('n')
        self.noop = 0.0 * self.n

        for _param, _grad in zip(self.model.params, self.model._grads):
            self.inc_updates[self.incs[_param]] = self.momentum * self.incs[_param] - self.learningrate * _grad 
            self.updates[_param] = _param + self.incs[_param]

        self._updateincs = theano.function([self.index], self.model._cost, 
                                     updates = self.inc_updates,
                givens = {self.model.inputs:self.data[self.index*self.batchsize:(self.index+1)*self.batchsize]})
        self._trainmodel = theano.function([self.n], self.noop, updates = self.updates)

    def step(self):
        cost = 0.0
        stepcount = 0.0
        for batch_index in self.rng.permutation(self.numbatches-1):
            stepcount += 1.0
            upd = self._updateincs(batch_index)
            # print upd
            cost = (1.0-1.0/stepcount)*cost + (1.0/stepcount)*upd#self._updateincs(batch_index)
            self._trainmodel(0)
            if self.normalizefilters:
                self.model.normalizefilters()

        # self.epochcount += 1
        # if self.verbose:
        #     print 'epoch: %d, cost: %f' % (self.epochcount, cost)
        return cost

    def reset_incs(self):
        for p in self.model.params:
            self.incs[p].set_value(numpy.zeros(p.get_value().shape, dtype=theano.config.floatX))

    def set_learningrate(self, v):
      self.learningrate.set_value(v)
