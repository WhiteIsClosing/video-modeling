################################################################################
# parameters defined by the user

## INPUT PATH
num_balls     = 3
project_path  = '/deep/u/kuanfang/optical-flow-pred/'
data_path     = project_path + 'data/bouncing_balls/'+ str(num_balls) + 'balls/' 
gc_path       = project_path + 'data/grammar_cells/' + str(num_balls) + 'balls/'

## DATA
image_suffix    = '.jpeg'   # suffix to load/store images
image_shape     = (16, 16)  # size of single channel images
numframes_train = 50000     # number of frames to train in each epoch
numframes_test  = 10000     # number of frames to validate in each epoch
seq_len         = 5         # number of frames in a sequence

## GRAMMAR CELL
numfac    = 80      # dimension of the feature factors
numvel    = 40      # dimension of the velocity units 
numvelfac = 40      # dimension of the velocity factors
numacc    = 20      # dimension of the acceleration units
numaccfac = 10      # dimension of the acceleration factors
numjerk   = 10      # dimension of the jerk units

seq_len_to_train    = seq_len     # sequence length to train
seq_len_to_predict  = seq_len     # sequence length to predict

## SOLVER PARAMETERS
corrupt_rate = 0.9      # corruption rate

lr_v = 1.e-3            # learning rate of pretraining the velocity model
lr_a = 1.e-3            # learning rate of pretraining the acceleration model
lr_t = 1.e-3            # learning rate of training the model

max_epoch_v = 1000      # maximum of epochs of pretraining the velocity
max_epoch_a = 1000      # maximum of epochs of pretraining the acceleration
max_epoch_t = 1000      # maximum of epochs of training the model

bs_v = 50               # batch size of pretraining the velocity
bs_a = 10               # batch size of pretraining the acceleration
bs_t = 10               # batch size of trainining the model

epoch_temp_save = 500   # epoch number to save temporal back-ups of the model

## OUTPUT PATH
models_path = 'models/'         # path to store the models
pred_path   = 'prediction/'     # path to store the predictions
vis_path    = 'visualization/'  # path to store the visualied results

## PREDICT AND VISUALIZATION
pred_len = 50                   # length of the prediction

################################################################################
# parameters depend on above settings

## DATA
frame_dim = image_shape[0] * image_shape[1]   # data dimension of each frame
seq_dim = frame_dim * seq_len                 # data dimension of each sequence
numseqs_train = numframes_train / seq_len     # number of sequence to train
numseqs_test = numframes_test / seq_len       # number of sequence to validate
