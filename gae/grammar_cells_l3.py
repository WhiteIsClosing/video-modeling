import numpy, pylab
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


class GrammarCellsL3(object):
    """
    3-layer grammar cells
    """
    def __init__(self, 
                    dimx, dimfacx, 
                    dimv, dimfacv, 
                    dima, dimfaca, 
                    dimj, 
                    seq_len_train, seq_len_predict, 
                    output_type='real', coststart=4, 
                    vis_corrupt_type="zeromask", vis_corrupt_level=0.0, 
                    numpy_rng=None, theano_rng=None):

        # hyper parameters
        ########################################################################
        """
        """
        self.dimx = dimx
        self.dimfacx = dimfacx
        self.dimv = dimv
        self.dimfacv = dimfacv
        self.dima = dima
        self.dimfaca = dimfaca
        self.dimj = dimj

        self.seq_len_train = seq_len_train
        self.seq_len_predict = seq_len_predict

        self.output_type = output_type
        self.coststart = coststart

        self.vis_corrupt_type = vis_corrupt_type
        self.vis_corrupt_level =\
            theano.shared(value=numpy.array([vis_corrupt_level]), 
                                                name='vis_corrupt_level')

        if not numpy_rng:  
            self.numpy_rng = dimpy.random.RandomState(1)
        else:
            self.numpy_rng = dimpy_rng
        if not theano_rng:  
            theano_rng = RandomStreams(1)
        else:
            self.theano_rng = theano_rng

        # parameters
        ########################################################################
        """
        """
        self.wfx_left = self.init_weight('wfx_left', (dimfacx, dimx)) 
        self.wfx_right = self.init_weight('wfx_right', (dimfacx, dimx))  
        self.wv = self.init_weight('wv', (dimv, dimfacx))  
        self.wfv_left = self.init_weight('wfv_left', (dimfacv, dimv)) 
        self.wfv_right = self.init_weight('wfv_right', (dimfacv, dimv))  
        self.wa = self.init_weight('wa', (dima, dimfacv))  
        self.wfa_left = self.init_weight('wfa_left', (dimfaca, dima)) 
        self.wfa_right = self.init_weight('wfa_right', (dimfaca, dima))  
        self.wj = self.init_weight('wj', (dimj, dimfaca))  
        self.bx = self.init_bias('bx', (dimx)) 
        self.bv = self.init_bias('bv', (dimv)) 
        self.ba = self.init_bias('ba', (dima)) 
        self.bj = self.init_bias('bj', (dimj)) 
        self.autonomy = theano.shared(value=numpy.array([0.5]).\
                        astype("float32"), name='autonomy') # TODO: init_bias
        self.params = [self.wfx_left, self.wxf_right, 
                        self.wv, self.wfv_left, self.wfv_right, 
                        self.wa, self.wfa_left, self.wfa_right, 
                        self.wj, self.bx, self.bv, self.ba, self.bj, 
                        self.autonomy]

        # layers 
        ########################################################################
        """
        """
        # initialization of the layers
        self.inputs = T.matrix(name='inputs') 

        self._xs = [None] * self.seq_len_predict
        self._vels = [None] * self.seq_len_predict
        self._accs = [None] * self.seq_len_predict
        self._jerks = [None] * self.seq_len_predict
        self._recons = [None] * self.seq_len_predict

        # extracting the input data
        for t in range(self.seq_len_predict):
            if t < self.seq_len_train:
                self._xs[t] = self.inputs[:, t*dimx:(t+1)*dimx]
            else:
                self._xs[t] = T.zeros((self._xs[0].shape[0], self.dimx)) 

            # if t>3:
            #     self._xs[t] = self.corrupt(self._xs[t])
            self._xs[t] = self.corrupt(self._xs[t])
            
            if t >= 0 and t <= 3:
                self._recons[t] = self._xs[t]

        for t in range(4, self.seq_len_predict):
            self._facx_left[t-4] = T.dot(self._recons[t-4], self.wfx_left)
            self._facx_right[t-4] = T.dot(self._recons[t-4], self.wfx_right)
            self._facx_left[t-3] = T.dot(self._recons[t-3], self.wfx_left)
            self._facx_right[t-3] = T.dot(self._recons[t-3], self.wfx_right)
            self._facx_left[t-2] = T.dot(self._recons[t-2], self.wfx_left)
            self._facx_right[t-2] = T.dot(self._recons[t-2], self.wfx_right)
            self._facx_left[t-1] = T.dot(self._recons[t-1], self.wfx_left)
            self._facx_right[t-1] = T.dot(self._recons[t-1], self.wfx_right)
            self._facx_left[t] = T.dot(self._recons[t], self.wfx_left)
            self._facx_right[t] = T.dot(self._recons[t], self.wfx_right)

            #re-infer current velocities v12 and v23: 
            self._prevel01 = T.dot(self._facx_left[t-4]*self._facx_right[t-3], self.wv)+self.bv
            self._prevel12 = T.dot(self._facx_left[t-3]*self._facx_right[t-2], self.wv)+self.bv
            self._prevel23 = T.dot(self._facx_left[t-2]*self._facx_right[t-1], self.wv)+self.bv
            self._prevel34 = T.dot(self._facx_left[t-1]*self._facx_right[t  ], self.wv)+self.bv

            #re-infer acceleration a123: 
            self._preacc012 = T.dot(T.dot(T.nnet.sigmoid(self._prevel01), self.wfv_left)*T.dot(T.nnet.sigmoid(self._prevel12), self.wfv_right), self.wa)+self.ba
            self._preacc123 = T.dot(T.dot(T.nnet.sigmoid(self._prevel12), self.wfv_left)*T.dot(T.nnet.sigmoid(self._prevel23), self.wfv_right), self.wa)+self.ba
            self._preacc234 = T.dot(T.dot(T.nnet.sigmoid(self._prevel23), self.wfv_left)*T.dot(T.nnet.sigmoid(self._prevel34), self.wfv_right), self.wa)+self.ba

            if t==4:
                self._prejerks[t-1] = T.dot(T.dot(T.nnet.sigmoid(self._preacc012), self.wfa_left)*T.dot(T.nnet.sigmoid(self._preacc123), self.wfa_right), self.wj)+self.bj

            #infer jerk as weighted sum of past and re-infered: 
            self._prejerks[t] = T.nnet.sigmoid(self.autonomy[0])*self._prejerks[t-1]+(1-T.nnet.sigmoid(self.autonomy[0]))*(T.dot(T.dot(T.nnet.sigmoid(self._preacc123), self.wfa_left)*T.dot(T.nnet.sigmoid(self._preacc234), self.wfa_right), self.wj)+self.bj)

            #fill in all remaining activations from top-level jerk and past: 
            self._accs[t] = T.nnet.sigmoid(T.nnet.sigmoid(self.autonomy[0])*(T.dot(T.dot(T.nnet.sigmoid(self._prejerks[t]), self.wj.T) * T.dot(self._preacc123, self.wfa_left), self.wfa_right.T) + self.ba) + (1.0-T.nnet.sigmoid(self.autonomy[0]))*self._preacc234)
            self._vels[t] = T.nnet.sigmoid(T.nnet.sigmoid(self.autonomy[0])*(T.dot(T.dot(self._accs[t], self.wa.T)*T.dot(self._prevel23,self.wfv_left), self.wfv_right.T)+self.bv) + (1.0-T.nnet.sigmoid(self.autonomy[0]))*self._prevel34)
            self._recons[t] = T.dot(T.dot(self._recons[t-1],self.wfx_left)*T.dot(self._vels[t], self.wv.T),self.wxf_right.T) + self.bx

        self._prediction = T.concatenate([pred[:,:self.dimx] for pred in self._recons], 1)
        self._notebook = T.concatenate([pred[:,self.dimx:] for pred in self._recons], 1)
        if self.output_type == 'binary':
            self._prediction_for_training = T.concatenate([T.nnet.sigmoid(pred[:,:self.dimx]) for pred in self._recons[self.coststart:self.seq_len_train]], 1)
        else:
            self._prediction_for_training = T.concatenate([pred[:,:self.dimx] for pred in self._recons[self.coststart:self.seq_len_train]], 1)

        print self.output_type
        if self.output_type == 'real':
            self._cost = T.mean((self._prediction_for_training - self.inputs[:,self.coststart*self.dimx:self.seq_len_train*self.dimx])**2)
        elif self.output_type == 'binary':
            self._cost = -T.mean(self.inputs[:,self.coststart*self.dimx:self.seq_len_train*self.dimx]*T.log(self._prediction_for_training) 
                                    + 
                                 (1.0-self.inputs[:,self.coststart*self.dimx:self.seq_len_train*self.dimx])*T.log(1.0-self._prediction_for_training))

        self._grads = T.grad(self._cost, self.params)

        self.prediction = theano.function([self.inputs], self._prediction)
        self.notebook = theano.function([self.inputs], self._notebook)
        self.vels = [theano.function([self.inputs], v) for v in self._vels[4:]]
        self.accs = [theano.function([self.inputs], a) for a in self._accs[4:]]
        self.jerks = [theano.function([self.inputs], j) for j in self._prejerks[4:]]
        self.cost = theano.function([self.inputs], self._cost)
        self.grads = theano.function([self.inputs], self._grads)
        def get_cudandarray_value(x):
            if type(x)==theano.sandbox.cuda.CudaNdarray:
                return numpy.array(x.__array__()).flatten()
            else:
                return x.flatten()
        self.grad = lambda x: numpy.concatenate([get_cudandarray_value(g) for g in self.grads(x)])

    def predict(self, seedframes, seq_len=10):
        # seedframs.shape = (1, 160)
        dimcases = seedframes.shape[0]
        frames_and_notes = [numpy.concatenate((seedframes[:,i*self.dimx:(i+1)*self.dimx], dimpy.zeros((dimcases, self.dimnote), dtype="float32")),1) for i in range(seedframes.shape[1]/self.dimx)] 
        for i in range(seedframes.shape[1]/self.dimx, seq_len):
            frames_and_notes.append(numpy.zeros((dimcases, self.dimnote+self.dimx),dtype="float32"))

        firstprejerk = theano.function([self._xs[0], self._xs[1], self._xs[2], self._xs[3]], self._prejerks[3])
        prejerk = firstprejerk(frames_and_notes[0], frames_and_notes[1], frames_and_notes[2], frames_and_notes[3])

        next_prediction_and_jerk = theano.function([self._xs[1], self._xs[2], self._xs[3], self._xs[4], self._prejerks[3]], T.concatenate((self._recons[4], self._prejerks[4]), 1))

        preds = numpy.concatenate((seedframes[:,:self.dimx*4], dimpy.zeros((dimcases,(seq_len-4)*self.dimx),dtype="float32")), 1)

        for t in range(4, seq_len):
            preds_notebook_jerks = next_prediction_and_jerk(frames_and_notes[t-3], frames_and_notes[t-2], frames_and_notes[t-1], frames_and_notes[t], prejerk)
            frames_and_notes[t][:,:] = preds_notebook_jerks[:,:self.dimx+self.dimnote]
            prejerk = preds_notebook_jerks[:,self.dimx+self.dimnote:]
            preds[:,t*self.dimx:(t+1)*self.dimx] = preds_notebook_jerks[:,:self.dimx]
        return preds

    def init_weight(self, name, size, val=.01):
        """
        Utility function to initialize theano shared weights.
        """
        return theano.shared(value = val*self.numpy_rng.normal(size=size)\
                      .astype(theano.config.floatX), name=self.name+':'+name)

    def init_bias(self, name, size, val=0.):
        """
        Utility function to initialize theano shared bias.
        """
        return theano.shared(value = val*numpy.ones(size,
                          dtype=theano.config.floatX), name=self.name+':'+name) 

    def corrupt(self, raw):
        if self.vis_corrupt_type=='zeromask':
            corrupted = theano_rng.binomial(size=self.raw.shape, 
                n=1, p=1.0-self.vis_corrupt_level, 
                dtype=theano.config.floatX) * self.raw
        elif self.vis_corrupt_type=='mixedmask':
            corrupted = theano_rng.binomial(size=raw.shape, 
                n=1, p=1.0-self.vis_corrupt_level/2, 
                dtype=theano.config.floatX) * raw
            corrupted = (1-theano_rng.binomial(size=corrupted.shape, 
                n=1, p=1.0-self.vis_corrupt_level/2, 
                dtype=theano.config.floatX)) * corrupted
        elif self.vis_corrupt_type=='gaussian':
            corrupted = theano_rng.normal(size=raw.shape, avg=0.0, 
            std=self.vis_corrupt_level, dtype=theano.config.floatX) + raw
        else:
            assert False, "vis_corrupt type not understood"
        return corrupted

    def set_params(self, new_params):
        """
        Set all values in self.params to new_params.
        """

        def inplace_update(x, new):
            x[...] = new
            return x

        params_counter = 0
        for p in self.params:
            pshape = p.get_value().shape
            pnum = numpy.prod(pshape)
            p.set_value(inplace_update(p.get_value(borrow=True),
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

    def normalize_filters(self):
        """
        """
        raise Exception('Not impleted yet. ')


