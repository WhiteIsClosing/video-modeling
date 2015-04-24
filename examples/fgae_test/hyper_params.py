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

frame_dim = image_shape[0] * image_shape[1]   # data dimension of each frame
seq_dim = frame_dim * seq_len                 # data dimension of each sequence
numseqs_train = numframes_train / seq_len     # number of sequence to train
numseqs_test = numframes_test / seq_len       # number of sequence to validate

## GRAMMAR CELL
dimdat  = frame_dim
dimfac  = 80     
dimmap  = 40    

batch_size  = 100
lr          = 1.e-2
max_epoch   = 10000
save_epoch  = 1000

## OUTPUT PATH
models_path = 'models/'         # path to store the models
pred_path   = 'prediction/'     # path to store the predictions
vis_path   = 'visualization/'     # path to store the predictions

## PREDICT AND VISUALIZATION
pred_len = 50                   # length of the prediction

max_plot  = 1000                # maximum number of time step to plot
th_of     = 0.1                 # threashold to draw optical flows
