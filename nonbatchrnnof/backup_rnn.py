import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class RNN(object):
    
    def __init__(self, dimx, dimy, dimh, batch_size, length):
        '''
        dimh :: dimension of the hidden layer
        dimx :: dimension of the input
        dimy :: dimension of output
        '''
        # parameters of the model
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimx, dimh)).astype(theano.config.floatX))
        self.Wh  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimh, dimh)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimh, dimy)).astype(theano.config.floatX))
        self.bh  = theano.shared(numpy.zeros(dimh, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(dimy, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(dimh, dtype=theano.config.floatX))

        # bundle
        self.params = [self.Wx, self.Wh, self.W, self.bh, self.b, self.h0]
        self.names  = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']

        self.inframes = T.matrix(name='inframes') 
        self.truth = T.matrix(name='truth') 

        # batch_size = self.inframes.shape[0]
        # length = self.inframes.shape[1] / dimx

        self.x = [None] * batch_size
        self.d = [None] * batch_size

        for idx in range(batch_size):
          self.x[idx] = self.inframes[idx, :].reshape((length, dimx))
          self.d[idx] = self.truth[idx, :].reshape((length, dimy))
        

        def recurrence(x_t, h_tm1):
          h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
          # s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
          s_t = T.dot(h_t, self.W) + self.b
          return [h_t, s_t]
        
        self.h = [None] * batch_size
        self.s = [None] * batch_size

        for idx in range(batch_size):
          [self.h[idx], self.s[idx]], _ = theano.scan(fn=recurrence, \
              sequences=self.x[idx], outputs_info=[self.h0, None], \
              n_steps=length)

        self.outframes = T.concatenate([self.s[idx].reshape((1, dimy * length)) \
                for idx in range(batch_size)], axis = 0)


        # self.cost = T.mean((self.truth[:, dimy:] - self.outframes[:, dimy:]) ** 2)
        self.cost = T.mean((self.truth - self.outframes) ** 2)

        # cost and grads and learning rate
        self.lr = T.scalar('lr') # learning rate
        self.grads = T.grad( self.cost, self.params )
        updates = OrderedDict(( p, p-self.lr*g ) for p, g in zip( self.params , self.grads))
        
        # theano functions
        self.predict = theano.function(inputs = [self.inframes], outputs=self.outframes)

        self.train = theano.function( inputs  = [self.inframes, self.truth, self.lr],
                                      outputs = self.cost,
                                      updates = updates )

        # self.getX = theano.function(inputs = [self.inframes], \
        #               outputs = self.x)
        # self.getD = theano.function(inputs = [self.truth], \
        #               outputs = self.d)
        # self.getH = theano.function(inputs = [self.inframes], \
        #               outputs = self.h)
        # self.getS = theano.function(inputs = [self.inframes], \
        #                 outputs = self.s)
        # self.getGrads = theano.function(inputs = [self.inframes, self.truth], \
        #                       outputs = self.grads)
        self.getCost = theano.function(inputs = [self.inframes, self.truth], \
                        outputs = self.cost)
                             

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
