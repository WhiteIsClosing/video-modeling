################################################################################
# parameters defined by the user

## INPUT PATH
project_path  = '/deep/u/kuanfang/video-modeling/'
data_path     = project_path + 'data/patch_data/translation.npy'

## DATA
image_suffix    = '.jpeg'   # suffix to load/store images
image_shape     = (16, 16)  # size of single channel images
numframes_train = 20000     # number of frames to train in each epoch
numframes_test  = 1000     # number of frames to validate in each epoch
seq_len         = 5         # number of frames in a sequence

frame_dim = image_shape[0] * image_shape[1]   # data dimension of each frame
seq_dim = frame_dim * seq_len                 # data dimension of each sequence
numseqs_train = numframes_train / seq_len     # number of sequence to train
numseqs_test = numframes_test / seq_len       # number of sequence to validate

## GRAMMAR CELL
size_dat = image_shape
size_tile = (4, 4)
dimdat  = frame_dim
dimtile  = size_tile[0] * size_tile[1]
dimfac  = 100     
dimmap  = 30    

bs          = 1000
lr          = dimtile * 2.e-2
max_epoch   = 10000
save_epoch  = 100

## OUTPUT PATH
models_path = 'models/'         # path to store the models
pred_path   = 'prediction/'     # path to store the predictions
vis_path   = 'visualization/'     # path to store the predictions

## PREDICT AND VISUALIZATION
pred_len = 50                   # length of the prediction

max_plot  = 1000                # maximum number of time step to plot
th_of     = 0.1                 # threashold to draw optical flows
