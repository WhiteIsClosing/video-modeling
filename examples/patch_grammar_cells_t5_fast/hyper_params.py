################################################################################
# parameters defined by the user

## INPUT PATH
num_balls     = 1
project_path  = '/deep/u/kuanfang/video-modeling/'
data_path     = project_path + 'data/patch_data/'

## DATA
image_suffix    = '.jpeg'   # suffix to load/store images
image_shape     = (16, 16)  # size of single channel images
numseqs_train   = 5000      # number of sequence to train
numseqs_test    = 1000      # number of sequence to validate
seq_len         = 5        # number of frames in a sequence

frame_dim = image_shape[0] * image_shape[1] # data dimension of each frame
seq_dim = frame_dim * seq_len               # data dimension of each sequence
numframes_train = numseqs_train*seq_len     # number of frames to train
numframes_test  = numseqs_test*seq_len     # number of frames to validate

## GRAMMAR CELL
numfac    = 160      # dimension of the feature factors
numvel    = 80      # dimension of the velocity units 
numvelfac = 80      # dimension of the velocity factors
numacc    = 40      # dimension of the acceleration units
numaccfac = 20      # dimension of the acceleration factors
numjerk   = 20      # dimension of the jerk units

seq_len_to_train    = seq_len     # sequence length to train
seq_len_to_predict  = seq_len     # sequence length to predict

corrupt_rate = 0.5      # corruption rate

## SOLVER
max_epoch_v = 500      # maximum of epochs of pretraining the velocity
max_epoch_a = 300      # maximum of epochs of pretraining the acceleration
max_epoch_t = 100      # maximum of epochs of training the model

lr_v = 1.e-2            # learning rate of pretraining the velocity model
lr_a = 1.e-2            # learning rate of pretraining the acceleration model
lr_t = 5e-1            # learning rate of training the model

bs_v = 1000               # batch size of pretraining the velocity
bs_a = 1000               # batch size of pretraining the acceleration
bs_t = 1000               # batch size of trainining the model

epoch_temp_save = 500   # epoch number to save temporal back-ups of the model

## OUTPUT PATH
models_path = 'models/'         # path to store the models
pred_path   = 'prediction/'     # path to store the predictions
vis_path    = 'visualization/'  # path to store the visualied results

## PREDICT AND VISUALIZATION
pred_len = 50                   # length of the prediction

max_plot  = 1000                # maximum number of time step to plot
th_of     = 0.1                 # threashold to draw optical flows

################################################################################
# parameters depend on above settings

## DATA
frame_dim = image_shape[0] * image_shape[1]   # data dimension of each frame
seq_dim = frame_dim * seq_len                 # data dimension of each sequence
numseqs_train = numframes_train / seq_len     # number of sequence to train
numseqs_test = numframes_test / seq_len       # number of sequence to validate
