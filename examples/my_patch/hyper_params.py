################################################################################
# INPUT PATH
project_path  = '/deep/u/kuanfang/video-modeling/'
data_path     = project_path + 'data/patch_data/'

# DATA
image_suffix    = '.jpeg'   # suffix to load/store images
image_shape     = (16, 16)  # size of single channel images
numseqs_train = 20000     # number of sequence to train
numseqs_test = 1000       # number of sequence to validate
seq_len         = 5         # number of frames in a sequence

frame_dim = image_shape[0] * image_shape[1]   # data dimension of each frame
seq_dim = frame_dim * seq_len                 # data dimension of each sequence
numframes_train = numseqs_train * seq_len     # number of frames to train in each epoch
numframes_test  = numseqs_test * seq_len     # number of frames to validate in each epoch

# GRAMMAR CELL
dimx    = frame_dim
dimfacx = 100
dimv    = 30
dimfacv = 30
dima    = 20
dimfaca = 20
dimj    = 10

corrupt_level = 0.5

# SOLVER
max_epoch_v = 1000
max_epoch_a = 1000
max_epoch   = 100

lr_v        = dimx * 1.e-3
lr_a        = dimv * 1.e-3
lr          = 8.e-2

bs_v        = 1000
bs_a        = 1000
bs          = 100

# OUTPUT PATH
models_path = 'models/'         # path to store the models
pred_path   = 'prediction/'     # path to store the predictions
vis_path   = 'visualization/'   # path to store the visualized results


## PREDICT AND VISUALIZATION
pred_len = 50                   # length of the prediction

max_plot  = 1000                # maximum number of time step to plot
th_of     = 0.1                 # threashold to draw optical flows
