import numpy
import time
import sys
import random
import theano
from time import clock

from hyper_params import *
sys.path.insert(0, project_path)
from utils.load import *
from utils.log import *

from gae.tiled_gated_autoencoder import *
from gae.solver import *

seed = 42
numpy.random.seed(seed)
random.seed(seed)

logInfo = LogInfo('LOG.txt')

# LOAD DATA

features_numpy = numpy.load(data_path).astype('float32')
assert numseqs_train + numseqs_test <= features_numpy.shape[0]
assert seq_len <= features_numpy.shape[1]

features_train_numpy = features_numpy[:numseqs_train, :seq_len*frame_dim] 
features_test_numpy =\
    features_numpy[numseqs_train:numseqs_train+numseqs_test,
                    :seq_len*frame_dim] 
print features_train_numpy.shape
print features_test_numpy.shape

# PREPROCESS
data_mean = features_train_numpy.mean()
data_std = features_train_numpy.std()
features_train_numpy -= data_mean
features_train_numpy /= data_std 
features_test_numpy -= data_mean
features_test_numpy /= data_std
# test_feature_beginnings = features_test_numpy[:,:frame_dim*3]
features_train_theano = theano.shared(features_train_numpy)

# CONCATENATE
features_train_numpy =\
    numpy.concatenate([features_train_numpy[i, 2*j*frame_dim:2*(j+1)*frame_dim]\
    [None,:]\
    for j in range(seq_dim/(frame_dim*2)) for i in range(numseqs_train)],0)

features_train_numpy = features_train_numpy[ \
    numpy.random.permutation(features_train_numpy.shape[0])]

numpy.save('features_train_numpy', features_train_numpy)

features_theano = theano.shared(features_train_numpy)

# INITIALIZATION
print 'start initialization ...'
tic = clock()

model = TiledGatedAutoencoder(
                            size_dat=size_dat,
                            size_tile=size_tile,
                            dimfac=dimfac,
                            dimmap=dimmap,
                            corrupt_type='zeromask', 
                            corrupt_level=0.3, 
                            )

toc = clock()
logInfo.mark('... initialization done with {} sec'.format(toc-tic))

solver = GraddescentMinibatch(model, features_theano, 
                                batch_size=bs, learning_rate=lr)

print 'start training ...'
round = 0
for epoch in xrange(max_epoch):
    tic_e = clock()
    cost = solver.step()
    toc_e = clock()
    logInfo.mark('# epoch: '+str(epoch) + '\tcost: ' + str(cost) + \
                '\tlearning_rate: ' + str(lr) + '\ttime: ' + str(toc_e-tic_e))
    if (epoch+1) % save_epoch == 0:
        model.save(models_path + 'model')
        # numpy.save(pred_path + 'recons', recons.flatten())
        logInfo.mark('Round ' + str(round) + '. Saved model @ ' + models_path)
        print 'train cost: {}'.format(model.f_cost(features_train_numpy))

