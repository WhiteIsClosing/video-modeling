### Train grammar-cells on bouncing balls

import sys
import numpy
import numpy.random
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from scipy import misc
from time import clock
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)

from hyper_params import *
sys.path.insert(0, project_path)
from utils.load import *
from utils.log import *

from grammar_cells.gatedAutoencoder import *
from grammar_cells.pgp3layer import *
from grammar_cells.solver import *


#LOG# 
################################################################################
logInfo = LogInfo('LOG.txt')


# LOAD DATA
################################################################################
tic = clock()
features_numpy = numpy.load(data_path + 'translation.npy').astype('float32')
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
features_train_theano = theano.shared(features_train_numpy)
features_test_numpy -= data_mean
features_test_numpy /= data_std
# test_feature_beginnings = features_test_numpy[:,:frame_dim*3]
toc = clock()
logInfo.mark('time of loading data: ' + str(toc - tic))


# PRETRAIN VELOCITY MODEL
################################################################################
logInfo.mark('[pretraining]')
logInfo.mark('<pretrain velocity model>')
pretrain_model_velocity = FactoredGatedAutoencoder(
                          numvisX=frame_dim,
                          numvisY=frame_dim,
                          numfac=numfac, 
                          nummap=numvel, 
                          output_type='real', 
                          corruption_type='none',#'zeromask', 
                          corruption_level=0.0,#0.3, 
                          numpy_rng=numpy_rng, 
                          theano_rng=theano_rng)

pretrain_features_velocity_numpy = numpy.concatenate([features_train_numpy[i, 2*j*frame_dim:2*(j+1)*frame_dim][None,:] \
  for j in range(seq_dim/(frame_dim*2)) for i in range(numseqs_train)],0)

pretrain_features_velocity_numpy = pretrain_features_velocity_numpy[ \
numpy.random.permutation(pretrain_features_velocity_numpy.shape[0])]

pretrain_features_velocity_theano = theano.shared(pretrain_features_velocity_numpy)

tic = clock()

pretrainer_velocity = \
GraddescentMinibatch(pretrain_model_velocity, pretrain_features_velocity_theano, batchsize=bs_v, learningrate=lr_v)
for epoch in xrange(max_epoch_v):
  tic_e = clock()
  cost = pretrainer_velocity.step()
  toc_e = clock()
  logInfo.mark('# epoch: '+str(epoch) + '\tcost: ' + str(cost) + '\tlearning_rate: ' + str(lr_v) + '\ttime: ' + str(toc_e-tic_e))

toc = clock()
logInfo.mark('time of pretraining the velocity model: ' + str(toc - tic))
pretrain_model_velocity.save(models_path + 'pretrain_vel')


# PRETRAIN ACCELERATION MODEL
################################################################################
logInfo.mark('<pretrain acceleration model>')
pretrain_model_acceleration = FactoredGatedAutoencoder(
                              numvisX=pretrain_model_velocity.nummap,
                              numvisY=pretrain_model_velocity.nummap,
                              numfac=numvelfac, 
                              nummap=numacc, 
                              output_type='real', 
                              corruption_type='none',#'zeromask', 
                              corruption_level=0.0,#0.3, 
                              numpy_rng=numpy_rng, 
                              theano_rng=theano_rng)

pretrain_features_acceleration_numpy = \
numpy.concatenate((pretrain_model_velocity.mappings(features_train_numpy[:, :2*frame_dim]), \
  pretrain_model_velocity.mappings(features_train_numpy[:, 1*frame_dim:3*frame_dim])),1)

pretrain_features_acceleration_theano = theano.shared(pretrain_features_acceleration_numpy)

tic = clock()

pretrainer_acceleration = \
GraddescentMinibatch(pretrain_model_acceleration, pretrain_features_acceleration_theano, batchsize=bs_a, learningrate=lr_a)
for epoch in xrange(max_epoch_a):
  tic_e = clock()
  cost = pretrainer_acceleration.step()
  toc_e = clock()
  logInfo.mark('# epoch: '+str(epoch) + '\tcost: ' + str(cost) + '\tlearning_rate: ' + str(lr_a) + '\ttime: ' + str(toc_e-tic_e))

toc = clock()

logInfo.mark('time of pretraining the acceleration model: ' + str(toc - tic))
logInfo.mark('\n')

pretrain_model_acceleration.save(models_path + 'pretrain_acc')


# INITIALIZATION
################################################################################
logInfo.mark('[Training]')

tic = clock()
model = Pgp3layer(numvis=frame_dim,
                  numnote=0,
                  numfac=numfac,
                  numvel=numvel,
                  numvelfac=numvelfac,
                  numacc=numacc,
                  numaccfac=numaccfac,
                  numjerk=numjerk,
                  seq_len_to_train=seq_len_to_train,
                  seq_len_to_predict=seq_len_to_predict,
                  output_type='real',
                  vis_corruption_type='zeromask',
                  vis_corruption_level=0.0,
                  numpy_rng=numpy_rng,
                  theano_rng=theano_rng)

model.wxf_left.set_value(numpy.concatenate(\
(pretrain_model_velocity.wxf.get_value()*0.5, \
numpy_rng.randn(model.numnote,model.numfac).astype("float32")*0.001),0))

model.wxf_right.set_value(numpy.concatenate( \
(pretrain_model_velocity.wyf.get_value()*0.5, \
numpy_rng.randn(model.numnote,model.numfac).astype("float32")*0.001),0))

model.wv.set_value(pretrain_model_velocity.whf.get_value().T)
model.wvf_left.set_value(pretrain_model_acceleration.wxf.get_value())
model.wvf_right.set_value(pretrain_model_acceleration.wyf.get_value())
model.wa.set_value(pretrain_model_acceleration.whf.get_value().T)
model.ba.set_value(pretrain_model_acceleration.bmap.get_value())
model.bv.set_value(pretrain_model_velocity.bmap.get_value())

model.bx.set_value(numpy.concatenate( \
(pretrain_model_velocity.bvisX.get_value(), \
numpy.zeros((model.numnote),dtype="float32"))))

model.autonomy.set_value(numpy.array([0.5], dtype="float32"))

toc = clock()
logInfo.mark('time of initializing the model: ' + str(toc - tic))


# TRAIN MODEL
################################################################################
logInfo.mark('<train the sequence model>')

trainer = GraddescentMinibatch(model, features_train_theano, batchsize=bs_t, learningrate=lr_t)

idx = 0;
while (1):
  tic = clock()
  
  model.vis_corruption_level.set_value(numpy.array([0.]).astype("float32"))
  for epoch in xrange(max_epoch_t/2):
    tic_e = clock()
    cost = trainer.step()
    toc_e = clock()
    logInfo.mark('# epoch: '+str(epoch) + '\tcost: '+str(cost) + '\tlearning_rate: ' + str(lr_t) + '\ttime: ' + str(toc_e-tic_e))
    #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))
    if (epoch+1) % epoch_temp_save == 0:
      model.save(models_path + 'model_temp')
      # pred_frames_train = model.predict(features_train_numpy, pred_len)
      # numpy.save(pred_path + 'pred_frames_train_temp', pred_frames_train)
      # pred_frames_test = model.predict(features_test_numpy, pred_len)
      # numpy.save(pred_path + 'pred_frames_test_temp', pred_frames_test)
      cost_test = model.cost(features_test_numpy)
      logInfo.mark('stored temperal model @ ' + models_path + 'cost_test: ' + str(cost_test))

  model.vis_corruption_level.set_value(numpy.array([corrupt_rate]).astype("float32"))
  for epoch in xrange(max_epoch_t/2):
    tic_e = clock()
    cost = trainer.step()
    toc_e = clock()
    logInfo.mark('# epoch: '+str(epoch) + '\tcost: '+str(cost) + '\tlearning_rate: ' + str(lr_t) + '\ttime: ' + str(toc_e-tic_e))
    #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))
    if (epoch+1) % epoch_temp_save == 0:
      model.save(models_path + 'model_temp')
      # pred_frames_train = model.predict(features_train_numpy, pred_len)
      # numpy.save(pred_path + 'pred_frames_train_temp', pred_frames_train)
      # pred_frames_test = model.predict(features_test_numpy, pred_len)
      # numpy.save(pred_path + 'pred_frames_test_temp', pred_frames_test)
      cost_test = model.cost(features_test_numpy)
      logInfo.mark('stored temperal model @ ' + models_path + 'cost_test: ' + str(cost_test))

  toc = clock()
  logInfo.mark('training round: ' + str(idx) + '\ttime: ' + str(toc-tic))

  # saving model
  model.save(models_path + 'model')
  model.save(models_path + 'model_' + str(idx))

  pred_frames_train = model.predict(features_train_numpy, pred_len)
  numpy.save(pred_path + 'pred_frames_train', pred_frames_train)
  pred_frames_test = model.predict(features_test_numpy, pred_len)
  numpy.save(pred_path + 'pred_frames_test', pred_frames_test)

  cost_test = model.cost(features_test_numpy)
  logInfo.mark('saved model @ ' + models_path + 'cost_test: ' + str(cost_test))

  idx = idx + 1


flog.close()
