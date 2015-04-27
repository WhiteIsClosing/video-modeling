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

from gae.grammar_cells_l3 import *
from gae.solver import *

numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
seed = 42
numpy.random.seed(seed)
random.seed(seed)

# theano.config.compute_test_value = 'warn' # debug
# theano.config.exception_verbosity='high' # debug

logInfo = LogInfo('LOG.txt')

# LOAD DATA
################################################################################
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
features_test_numpy -= data_mean
features_test_numpy /= data_std
# test_feature_beginnings = features_test_numpy[:,:frame_dim*3]
features_train_theano = theano.shared(features_train_numpy)


# PRETRAIN VELOCITY MODEL
################################################################################
logInfo.mark('[pretraining]')
logInfo.mark('<pretrain velocity model>')

model_pretrain_vel = GatedAutoencoder(
                        dimdat=dimx, 
                        dimfac=dimfacx,
                        dimmap=dimv,
                        output_type='real', 
                        corrupt_type='none',#'zeromask', 
                        corrupt_level=0.0,#0.3, 
                        numpy_rng=numpy_rng, 
                        theano_rng=theano_rng)

logInfo.mark('... initialization done')

# prepare
features_vel_numpy =\
    numpy.concatenate([features_train_numpy[i, 2*j*frame_dim:2*(j+1)*frame_dim]\
    [None,:]\
    for j in range(seq_dim/(frame_dim*2)) for i in range(numseqs_train)],0)

features_vel_numpy = features_vel_numpy[ \
    numpy.random.permutation(features_vel_numpy.shape[0])]

features_vel_theano = theano.shared(features_vel_numpy)

solver_pretrain_vel = GraddescentMinibatch(model_pretrain_vel,
                                            features_vel_theano, 
                                            batch_size=bs_v, learning_rate=lr_v)

# train
tic = clock()

for epoch in xrange(max_epoch_v):
    tic_e = clock()
    cost = solver_pretrain_vel.step()
    toc_e = clock()
    logInfo.mark('# epoch: '+str(epoch) + '\tcost: ' + str(cost) + \
                '\tlearning_rate: ' + str(lr) + '\ttime: ' + str(toc_e-tic_e))

toc = clock()
logInfo.mark('time of pretraining the velocity model: ' + str(toc - tic))

# save
model_pretrain_vel.save(models_path + 'model_vel')
logInfo.mark('saved model_pretrain_vel @ ' + models_path)


# PRETRAIN ACCELERATION MODEL
################################################################################
logInfo.mark('<pretrain acceleration model>')
model_pretrain_acc = GatedAutoencoder(
                        dimdat=dimv, 
                        dimfac=dimfacv,
                        dimmap=dima,
                        output_type='real', 
                        #corrupt_type='zeromask', 
                        #corrupt_level=0.3, 
                        numpy_rng=numpy_rng, 
                        theano_rng=theano_rng)
# prepare
features_acc_numpy = \
    numpy.concatenate((
    model_pretrain_vel.f_map(features_train_numpy[:, 0*frame_dim:2*frame_dim]), 
    model_pretrain_vel.f_map(features_train_numpy[:, 1*frame_dim:3*frame_dim])),
    axis=1)
# features_acc_numpy = \
# numpy.concatenate((
#     model_pretrain_vel.f_map(
#     features_train_numpy[i, 2*j*frame_dim:2*(j+1)*frame_dim]),
#     model_pretrain_vel.f_map(
#     features_train_numpy[i, (2*j+1)*frame_dim:(2*(j+1)+1)*frame_dim])),
#     axis=1)

# features_acc_numpy = features_acc_numpy[ \
#     numpy.random.permutation(features_acc_numpy.shape[0])]

features_acc_theano = theano.shared(features_acc_numpy)

solver_pretrain_acc = GraddescentMinibatch(model_pretrain_acc, 
                                            features_acc_theano, 
                                            batch_size=bs_a, learning_rate=lr_a)

# train
tic = clock()

for epoch in xrange(max_epoch_a):
    tic_e = clock()
    cost = solver_pretrain_acc.step()
    toc_e = clock()
    logInfo.mark('# epoch: '+str(epoch) + '\tcost: ' + str(cost) + \
                '\tlearning_rate: ' + str(lr) + '\ttime: ' + str(toc_e-tic_e))

toc = clock()
logInfo.mark('time of pretraining the accocity model: ' + str(toc - tic))

# save
model_pretrain_acc.save(models_path + 'model_acc')
logInfo.mark('saved model_pretrain_acc @ ' + models_path)


# INITIALIZATION
################################################################################
logInfo.mark('[Training]')

model = GrammarCellsL3(\
                        dimx=dimx, 
                        dimfacx=dimfacx, 
                        dimv=dimv, 
                        dimfacv=dimfacv, 
                        dima=dima, 
                        dimfaca=dimfaca, 
                        dimj=dimj, 
                        seq_len=seq_len, 
                        output_type='real', 
                        vis_corrupt_type="zeromask", 
                        vis_corrupt_level=0.0, 
                        numpy_rng=None, theano_rng=None)

model.wfx_left.set_value(model_pretrain_vel.wfd_left.get_value())
model.wfx_right.set_value(model_pretrain_vel.wfd_right.get_value())
model.bx.set_value(model_pretrain_vel.bd.get_value())
model.wv.set_value(model_pretrain_vel.wmf.get_value())
model.bv.set_value(model_pretrain_vel.bm.get_value())

model.wfv_left.set_value(model_pretrain_acc.wfd_left.get_value())
model.wfv_right.set_value(model_pretrain_acc.wfd_right.get_value())
model.wa.set_value(model_pretrain_acc.wmf.get_value())
model.ba.set_value(model_pretrain_acc.bm.get_value())

# model.autonomy.set_value(numpy.array([0.5], dtype="float32"))

solver = GraddescentMinibatch(model, features_train_theano, 
                                batch_size=bs, learning_rate=lr)

round = 0
while (1):
    model.vis_corrupt_level.set_value(numpy.array([0.]).astype("float32"))
    for epoch in xrange(max_epoch):
        tic_e = clock()
        cost = solver.step()
        toc_e = clock()
        logInfo.mark('# epoch: '+str(epoch) + '\tcost: ' + str(cost) + \
            '\tlearning_rate: ' + str(lr) + '\ttime: ' + str(toc_e-tic_e))

    cost_test = model.f_cost(features_test_numpy)
    model.save(models_path + 'model')

    #
    model.vis_corrupt_level.set_value(numpy.array([corrupt_level]).\
                                                astype("float32"))
    for epoch in xrange(max_epoch):
        tic_e = clock()
        cost = solver.step()
        toc_e = clock()
        logInfo.mark('# epoch: '+str(epoch) + '\tcost: ' + str(cost) + \
            '\tlearning_rate: ' + str(lr) + '\ttime: ' + str(toc_e-tic_e))

    cost_test = model.f_cost(features_test_numpy)
    model.save(models_path + 'model_corrupted')
    # preds_train = model.f_preds(features_train_numpy)
    # preds_test = model.f_preds(features_test_numpy)
    # numpy.save(pred_path + 'preds_train', preds_train)
    # numpy.save(pred_path + 'preds_test', preds_test)
    logInfo.mark('Round ' + str(round) + '\tcost_test: ' + str(cost_test))
    logInfo.mark('Saved model @ ' + models_path)

    round += 1

