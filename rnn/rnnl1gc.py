import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class RNNL1GC(object):
    
    def __init__(self, dimx, dimy, dimh, dimvel):
        '''
        dimx :: dimension of the input
        dimvel :: dimension of velocity mapping unit in the grammar-cell
        dimh :: dimension of the hidden layer
        dimy :: dimension of output
        '''

        # parameters of the model
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimx, dimh)).astype(theano.config.floatX))
        self.Wvel  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimvel, dimh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimh, dimh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimh, dimy)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(dimh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(dimy, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(dimh, dtype=theano.config.floatX))

        # bundle
        self.params = [self.Wx, self.Wvel, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['Wx', 'Wvel','Wh', 'W', 'bh', 'b', 'h0']
        x = T.matrix(name='x') 
        vels = T.matrix(name='vels') # input features
        d = T.matrix(name='d') # input ground truth

        def recurrence(x_t, vel_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(vel_t, self.Wvel)\
                   + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.dot(h_t, self.W) + self.b
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=[x, vels], outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        cost = T.mean((d[1:, :] - s[1:, :]) ** 2)

        # cost and gradients and learning rate
        lr = T.scalar('lr') # learning rate
        gradients = T.grad( cost, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.predict = theano.function(inputs=[x, vels], outputs=s)

        self.train = theano.function( inputs  = [x, vels, d, lr],
                                      outputs = cost,
                                      updates = updates )

        # self.normalize = theano.function( inputs = [],
        #                  updates = {self.emb:\
        #                  self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})
        self.getCost = theano.function(inputs = [x, vels, d], \
                        outputs = cost)

    def updateparams(self, newparams):
        def inplaceupdate(x, new):
            x[...] = new
            return x

        paramscounter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplaceupdate(p.get_value(borrow=True), \
              newparams[paramscounter:paramscounter+pnum].reshape(*pshape)), \
              borrow=True)
            paramscounter += pnum 

    def get_params(self):
        return numpy.concatenate([p.get_value(borrow=False).flatten() \
                for p in self.params])

    def save(self, filename):
        numpy.save(filename, self.get_params())

    def load(self, filename):
        self.updateparams(numpy.load(filename))
