import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class Batch_RNNL1(object):
    
    def __init__(self, dimx, dimy, dimh, frame_dim, seq_len):
        '''
        dimh :: dimension of the hidden layer
        dimx :: dimension of the input
        dimy :: dimension of output
        frame_dim :: dimension of the frame
        seq_len :: length of the input sequence
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
        # self.h0  = theano.shared(numpy.zeros(dimh, dtype=theano.config.floatX))
        self.h0  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (1, dimh)).astype(theano.config.floatX))
        self.frame_dim = frame_dim
        self.seq_len = seq_len

        # bundle
        self.params = [self.Wx, self.Wh, self.W, self.bh, self.b, self.h0 ]
        self.names  = ['Wx', 'Wh', 'W', 'bh', 'b', 'h0']
        x = T.matrix(name='x') 
        d = T.matrix(name='d') # output: optical flow
        
        h = [None] * seq_len
        s = [None] * seq_len
        din = [None] * seq_len

        self.ones = 0 * x[:, 0] + 1
        self.BH = T.outer(self.ones, self.bh)
        self.B = T.outer(self.ones, self.b)
        self.H0 = T.outer(self.ones, self.h0)

        def recurrence(x_t, h_tm1):
          h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.BH)
          # s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
          s_t = T.dot(h_t, self.W) + self.B
          return [h_t, s_t]

        #[h, s], _ = theano.scan(fn=recurrence, \
        #    sequences=x, outputs_info=[self.h0, None], \
        #    n_steps=x.shape[0])

        for t in range(seq_len):
          din[t] = x[:, t*frame_dim:(t+1)*frame_dim]

        for t in range(seq_len):
          if t == 0:
            [h_t, s_t] = recurrence(din[t], self.H0)
          else:
            [h_t, s_t] = recurrence(din[t], h[t-1])
          h[t] = h_t
          s[t] = s_t

        y = T.concatenate([s[t] for t in range(seq_len)], axis = 1)


        cost = T.mean((d[:, 1*2*frame_dim:] - y[:, 1*2*frame_dim:]) ** 2)

        # cost and gradients and learning rate
        lr = T.scalar('lr') # learning rate
        gradients = T.grad( cost, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.get_ones = theano.function(inputs=[x], outputs=self.ones)
        self.get_s = theano.function(inputs=[x], outputs=s)
        self.get_H0 = theano.function(inputs=[x], outputs=self.H0)
        # self.d = theano.function(inputs=[x, d], outputs=d)
        # self.y = theano.function(inputs=[x, d], outputs=y)

        self.predict = theano.function(inputs=[x], outputs=y)

        self.train = theano.function( inputs  = [x, d, lr],
                                      outputs = cost,
                                      updates = updates )

        # self.normalize = theano.function( inputs = [],
        #                  updates = {self.emb:\
        #                  self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})
        self.getCost = theano.function(inputs = [x, d], \
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
