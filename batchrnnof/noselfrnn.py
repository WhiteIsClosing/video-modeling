import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class rnn(object):
    
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
        self.params = [self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']

        input_frames = T.matrix(name='input_frames') 
        groud_truth = T.matrix(name='groud_truth') 

        # batch_size = input_frames.shape[0]
        # length = input_frames.shape[1] / dimx

        x = [None] * batch_size
        d = [None] * batch_size

        for idx in range(batch_size):
          x[idx] = input_frames[idx, :].reshape((length, dimx))
          d[idx] = groud_truth[idx, :].reshape((length, dimy))
        
        output_frames = theano.shared(value = 0.0*numpy.ones((batch_size, dimy * length), \
                          dtype=theano.config.floatX), name='output_frames')


        def recurrence(x_t, h_tm1):
          h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
          s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
          return [h_t, s_t]
        
        h = [None] * batch_size
        s = [None] * batch_size

        for idx in range(batch_size):
          [h[idx], s[idx]], _ = theano.scan(fn=recurrence, \
              sequences=x[idx], outputs_info=[self.h0, None], \
              n_steps=batch_size)

        preds = T.concatenate([s[idx].reshape((1, idx * length)) \
                for idx in range(batch_size)], axis = 0)


        cost = T.mean((groud_truth - preds) ** 2)

        # cost and gradients and learning rate
        lr = T.scalar('lr') # learning rate
        gradients = T.grad( cost, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        predict = theano.function(inputs=[input_frames], outputs=preds)

        self.train = theano.function( inputs  = [input_frames, groud_truth, lr],
                                      outputs = cost,
                                      updates = updates )

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
