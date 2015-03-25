import theano
import numpy
import os

from theano import tensor as T
from collections import OrderedDict

class rnn(object):
    
    def __init__(self, dimh, dimy, ne, dimx, maxT):
        '''
        dimh :: dimension of the hidden layer
        dimx :: dimension of the input
        dimy :: dimension of output
        maxT :: length of sequence
        '''
        # parameters of the model
        self.Wx  = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0,\
                   (dimx * maxT, dimh)).astype(theano.config.floatX))
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
        # idxs = T.imatrix() # as many columns as context window size/lines as words in the sentence
        # x = self.emb[idxs].reshape((idxs.shape[0], dimx*maxT))
        self.inputs = T.matrix(name='inputs') 
        self.x = T.matrix(name='x') 
        self.y = T.matrix(name='y') # output: optical flow

        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.Wx) + T.dot(h_tm1, self.Wh) + self.bh)
            s_t = T.nnet.softmax(T.dot(h_t, self.W) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence, \
            sequences=x, outputs_info=[self.h0, None], \
            n_steps=x.shape[0])

        # p_y_given_x_lastword = s[-1,0,:]
        # p_y_given_x_sentence = s[:,0,:]
        # y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        self.cost = T.mean((y[1:, :] - s[1:, :]) ** 2)

        # cost and gradients and learning rate
        lr = T.scalar('lr') # learning rate
        # nll = -T.mean(T.log(p_y_given_x_lastword)[y])
        gradients = T.grad( cost, self.params )
        updates = OrderedDict(( p, p-lr*g ) for p, g in zip( self.params , gradients))
        
        # theano functions
        self.classify = theano.function(inputs=[idxs], outputs=y_pred)

        self.train = theano.function( inputs  = [idxs, y, lr],
                                      outputs = cost,
                                      updates = updates )

        # self.normalize = theano.function( inputs = [],
        #                  updates = {self.emb:\
        #                  self.emb/T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0,'x')})

    def save(self, folder):   
        for param, name in zip(self.params, self.names):
            numpy.save(os.path.join(folder, name + '.npy'), param.get_value())
