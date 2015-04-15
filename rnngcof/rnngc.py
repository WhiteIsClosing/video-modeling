import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class RNNGC(object):
    
    def __init__(self, dimx, dimy, dimh1, dimh2, dimvel, gc):
        '''
        dimx :: dimension of the input
        dimh1 :: dimension of the hidden layer
        dimh2 :: dimension of the mid-level layer
        dimy :: dimension of output
        dimvel :: dimension of velocity mapping unit in the grammar-cell
        gc :: grammar cell parameters
        '''

        # parameters of the model
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimx, dimh1)).astype(theano.config.floatX))
        self.Wh1  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimh1, dimh2)).astype(theano.config.floatX))
        self.Wh2  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimh2, dimh2)).astype(theano.config.floatX))
        self.Wvel  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimvel, dimh2)).astype(theano.config.floatX))
        self.W   = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimh2, dimy)).astype(theano.config.floatX))
        self.bh1  = theano.shared(numpy.zeros(dimh1, dtype=theano.config.floatX))
        self.bh2  = theano.shared(numpy.zeros(dimh2, dtype=theano.config.floatX))
        self.b   = theano.shared(numpy.zeros(dimy, dtype=theano.config.floatX))
        self.h0  = theano.shared(numpy.zeros(dimh2, dtype=theano.config.floatX))
        self.gc = gc

        # bundle
        self.params = [self.Wx, self.Wh1, self.Wh2, self.Wvel, self.W, self.bh1, self.bh2, self.b, self.h0 ]
        self.names  = ['Wx', 'Wh1', 'Wh2', 'Wvel', 'W', 'bh1', 'bh2', 'b', 'h0']
        x = T.matrix(name='x') # input features
        # z = T.matrix(name='h1') # mid-level layer
        d = T.matrix(name='d') # input ground truth

        # x_left_right = numpy.concatenate((x[:-1], x[1:]), axis=1)
        x_left = T.concatenate((T.zeros((1, x.shape[1])), x[:-1]), axis=0)
        x_right = x

        def getVel(frame_left, frame_right):
          factor_left = T.dot(frame_left, gc.wxf_left)
          factor_right = T.dot(frame_right, gc.wxf_right)
          return T.nnet.sigmoid(T.dot(factor_left*factor_right, gc.wv)+gc.bv)

        # x_left_t = x_left[0, :]
        # x_right_t = x_right[0, :]

        # frame_feature = T.nnet.sigmoid(T.dot(x_right_t, self.Wx) + self.bh1)
        # vel = getVel(x_left_t, x_right_t)
        # h1_t = T.concatenate([frame_feature, vel])#, axis=1)

        # print frame_feature.shape
        # print vel.shape
        # print h1_t.shape

        def recurrence(x_left_t, x_right_t, h2_tm1):
            h1_t = T.nnet.sigmoid(T.dot(x_right_t, self.Wx) + self.bh1)
            vel_t = getVel(x_left_t, x_right_t)
            h2_t = T.nnet.sigmoid(T.dot(h1_t, self.Wh1) + T.dot(vel_t, self.Wvel) + T.dot(h2_tm1, self.Wh2) + self.bh2)
            # s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            s_t = T.dot(h2_t, self.W) + self.b
            return [h2_t, s_t]

        [h2, s], _ = theano.scan(fn=recurrence, \
            sequences=[x_left, x_right], outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        cost = T.mean((d[1:, :] - s[1:, :]) ** 2)

        # cost and gradients and learning rate
        lr = T.scalar('lr') # learning rate
        gradients = T.grad( cost, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.predict = theano.function(inputs=[x], outputs=s)

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
