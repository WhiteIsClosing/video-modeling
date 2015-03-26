### Train grammar-cells on bouncing balls

import pylab
import numpy
import numpy.random
import gatedAutoencoder
import pgp3layer 
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
numpy_rng  = numpy.random.RandomState(1)
theano_rng = RandomStreams(1)
from scipy import misc
from time import clock

from hyperParams import *
from load import loadFromImg
from optimizer import GraddescentMinibatch

# LOAD DATA
train_features_numpy, test_features_numpy, numtrain, numtest, \
data_mean, data_std = loadFromImg()
train_features_theano = theano.shared(train_features_numpy)


# LOGGING HYPER PARAMETERS
flog = open('LOG.txt', 'w')
flog.write('[Model Parameters] \n')
flog.write('image_suffix: ' +  image_suffix + '\n')
flog.write('image_shape: ' + str(image_shape) + '\n')
flog.write('trainframes: ' + str(trainframes) + '\n')
flog.write('testframes: ' + str(testframes) + '\n')
flog.write('numframes: ' + str(numframes) + '\n')
flog.write('numframes_to_train_: ' + str(numframes_to_train_) + '\n')
flog.write('numframes_to_predict_: ' + str(numframes_to_predict_) + '\n')
flog.write('\n')
flog.write('numfac: ' + str(numfac_) + '\n')
flog.write('numvel: ' + str(numvel_) + '\n')
flog.write('numvelfac: ' + str(numvelfac_) + '\n')
flog.write('numacc: ' + str(numacc_) + '\n')
flog.write('numaccfac: ' + str(numaccfac_) + '\n')
flog.write('numjerk: ' + str(numjerk_) + '\n')
flog.write('\n')
flog.flush()


# PRETRAIN VELOCITY MODEL
print 'pretraining velocity model ...'
flog.write('[Pretraining] \n')
pretrainmodel_velocity = gatedAutoencoder.FactoredGatedAutoencoder(
                          numvisX=frame_len,
                          numvisY=frame_len,
                          numfac=numfac_, 
                          nummap=numvel_, 
                          output_type='real', 
                          corruption_type='none',#'zeromask', 
                          corruption_level=0.0,#0.3, 
                          numpy_rng=numpy_rng, 
                          theano_rng=theano_rng)

pretrain_features_velocity_numpy = numpy.concatenate( \
  [train_features_numpy[i, 2*j*frame_len:2*(j+1)*frame_len][None,:] \
  for j in range(seq_len/(frame_len*2)) for i in range(numtrain)],0)

pretrain_features_velocity_numpy = pretrain_features_velocity_numpy[ \
numpy.random.permutation(pretrain_features_velocity_numpy.shape[0])]

pretrain_features_velocity_theano = theano.shared(pretrain_features_velocity_numpy)

tic = clock()

pretrainer_velocity = \
  GraddescentMinibatch(pretrainmodel_velocity, pretrain_features_velocity_theano, batchsize=50, learningrate=1e-10)
for epoch in xrange(10):
    cost = pretrainer_velocity.step()

pretrainer_velocity = \
  GraddescentMinibatch(pretrainmodel_velocity, pretrain_features_velocity_theano, batchsize=50, learningrate=1e-5)
for epoch in xrange(50000):
    cost = pretrainer_velocity.step()
    if (cost <= 0.8): # set a threshold
        break

toc = clock()
flog.write('<pretrain velocity> ' + '\tcost: ' + str(cost) + '\t\ttime: ' + str(toc-tic) + '\n')
flog.flush()
print '... done'

# PRETRAIN ACCELERATION MODEL
print 'pretraining acceleration model ...'
pretrainmodel_acceleration = gatedAutoencoder.FactoredGatedAutoencoder(
                              numvisX=pretrainmodel_velocity.nummap,
                              numvisY=pretrainmodel_velocity.nummap,
                              numfac=numvelfac_, 
                              nummap=numacc_, 
                              output_type='real', 
                              corruption_type='none',#'zeromask', 
                              corruption_level=0.0,#0.3, 
                              numpy_rng=numpy_rng, 
                              theano_rng=theano_rng)

pretrain_features_acceleration_numpy = \
numpy.concatenate((pretrainmodel_velocity.mappings(train_features_numpy[:, :2*frame_len]), \
  pretrainmodel_velocity.mappings(train_features_numpy[:, 1*frame_len:3*frame_len])),1)

pretrain_features_acceleration_theano = theano.shared(pretrain_features_acceleration_numpy)

tic = clock()

pretrainer_acceleration = \
GraddescentMinibatch(pretrainmodel_acceleration, pretrain_features_acceleration_theano, batchsize=10, learningrate=1e-10)
for epoch in xrange(10):
  cost = pretrainer_acceleration.step()

pretrainer_acceleration = \
GraddescentMinibatch(pretrainmodel_acceleration, pretrain_features_acceleration_theano, batchsize=10, learningrate=1e-5)
for epoch in xrange(2000):
  cost = pretrainer_acceleration.step()

toc = clock()

  #pylab.imshow(pretrainmodel_acceleration.mappings(pretrain_features_acceleration_numpy[:200]))
  #pylab.show(); pylab.draw()

flog.write('<pretrain acceleration> ' + '\tcost: ' + str(cost) + '\t\ttime: ' + str(toc-tic) + '\n')
flog.write('\n')
flog.flush()
print '... done'


print 'training sequence model ...'
flog.write('[Training] \n')
flog.flush()
tic = clock()
model = pgp3layer.Pgp3layer(numvis=frame_len,
                          numnote=0,
                          numfac=numfac_,
                          numvel=numvel_,
                          numvelfac=numvelfac_,
                          numacc=numacc_,
                          numaccfac=numaccfac_,
                          numjerk=numjerk_,
                          numframes_to_train=numframes_to_train_,
                          numframes_to_predict=numframes_to_predict_,
                          output_type='real',
                          vis_corruption_type='zeromask',
                          vis_corruption_level=0.0,
                          numpy_rng=numpy_rng,
                          theano_rng=theano_rng)

model.wxf_left.set_value(numpy.concatenate(\
(pretrainmodel_velocity.wxf.get_value()*0.5, \
numpy_rng.randn(model.numnote,model.numfac).astype("float32")*0.001),0))

model.wxf_right.set_value(numpy.concatenate( \
(pretrainmodel_velocity.wyf.get_value()*0.5, \

numpy_rng.randn(model.numnote,model.numfac).astype("float32")*0.001),0))
model.wv.set_value(pretrainmodel_velocity.whf.get_value().T)
model.wvf_left.set_value(pretrainmodel_acceleration.wxf.get_value())
model.wvf_right.set_value(pretrainmodel_acceleration.wyf.get_value())
model.wa.set_value(pretrainmodel_acceleration.whf.get_value().T)
model.ba.set_value(pretrainmodel_acceleration.bmap.get_value())
model.bv.set_value(pretrainmodel_velocity.bmap.get_value())

model.bx.set_value(numpy.concatenate( \
(pretrainmodel_velocity.bvisX.get_value(), \
numpy.zeros((model.numnote),dtype="float32"))))

model.autonomy.set_value(numpy.array([0.5], dtype="float32"))

toc = clock()
flog.write('<initialization of the model>\t\ttime: ' + str(toc-tic) + '\n')
flog.flush()
print '... done'


# TRAIN MODEL
print 'TRAIN MODEL ...'
models_root = 'models/'
predictions_root = 'predictions/'
cost = 0.;

trainer = GraddescentMinibatch(model, train_features_theano, batchsize=10, learningrate=1e-10)
for epoch in xrange(5):
    trainer.step()

idx = 0;
while (1):
    tic = clock()

    trainer = GraddescentMinibatch(model, train_features_theano, batchsize=10, learningrate=1e-4)
    # for epoch in xrange(500):
    #     trainer.step()
        #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))
        #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))
        #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))
        #plot(train_features_numpy[0].flatten())
        #plot(model.predict(train_features_numpy[[0]], 100).flatten())
    
    model.vis_corruption_level.set_value(numpy.array([0.]).astype("float32"))
    for epoch in xrange(2500):
        cost = trainer.step()
        #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))

    model.vis_corruption_level.set_value(numpy.array([0.5]).astype("float32"))
    for epoch in xrange(2500):
        cost = trainer.step()
        #model.autonomy.set_value(numpy.array([0.5], dtype="float32"))

    print 'Saving the parameters ...'

    model.save(models_root + 'param_bb_' + str(idx))
    prediction_train = model.predict(train_features_numpy, 50)
    prediction_test = model.predict(test_features_numpy, 50)
    numpy.save(predictions_root + 'train/' + 'prediction' + str(idx), prediction_train)
    numpy.save(predictions_root + 'test/' + 'prediction' + str(idx), prediction_test)

    toc = clock()
    flog.write('training round: ' + str(idx) + '\tcost: ' + str(cost) + '\ttime: ' + str(toc-tic) + '\n')
    flog.flush()

    idx = idx + 1


flog.close()
print '... done'
