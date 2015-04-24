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

from gae.fgae import *
from gae.solver import *

seed = 42
numpy.random.seed(seed)
random.seed(seed)

logInfo = LogInfo('LOG.txt')

# LOAD DATA
tic = clock()

features_train_numpy = \
    loadFrames(data_path + 'train/', image_shape, numframes_train, seq_len)
features_test_numpy = \
    loadFrames(data_path + 'test/', image_shape, numframes_test, seq_len)

# PREPROCESS
data_mean = features_train_numpy.mean()
data_std = features_train_numpy.std()
features_train_numpy -= data_mean
features_train_numpy /= data_std 
train_features_theano = theano.shared(features_train_numpy)
features_test_numpy -= data_mean
features_test_numpy /= data_std
test_feature_beginnings = features_test_numpy[:,:frame_dim*3]

features_train_numpy =\
    numpy.concatenate([features_train_numpy[i, 2*j*frame_dim:2*(j+1)*frame_dim]\
    [None,:]\
    for j in range(seq_dim/(frame_dim*2)) for i in range(numseqs_train)],0)

features_train_numpy = features_train_numpy[ \
    numpy.random.permutation(features_train_numpy.shape[0])]

features_theano = theano.shared(features_train_numpy)

solver = GraddescentMinibatch(model, features_theano, 
                                batch_size=batch_size, learning_rate=lr)

# INITIALIZATION
model = FGAE(dimdat=dimdat,
                dimfac=dimfac,
                dimmap=dimmap,
                mode='reconstruct')

logInfo.mark('... initialization done')

round = 0
for epoch in xrange(max_epoch):
    tic_e = clock()
    cost = solver.step()
    toc_e = clock()
    logInfo.mark('# epoch: '+str(epoch) + '\tcost: ' + str(cost) + \
                '\tlearning_rate: ' + str(lr) + '\ttime: ' + str(toc_e-tic_e))
    if (epoch+1) % save_epoch == 0:
        recons = model.predict(features_train_numpy)
        model.save(models_path + 'model')
        numpy.save(pred_path + 'recons', recons.flatten())
        logInfo.mark('Round ' + str(round) + '. Saved model @ ' + models_path)
    round += 1

