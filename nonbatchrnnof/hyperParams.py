# DATA SIZE
data_path = '../data/bouncing_balls/1balls/' 
image_suffix = '.jpeg'
image_shape = (16, 16) # single channel images
numframes_train = 5000
numframes_test = 1000
seq_len = 5

# Data size paramters according to user configurations:
frame_dim = image_shape[0] * image_shape[1] # single channel images
seq_dim = frame_dim * seq_len
numseqs_train = numframes_train / seq_len
numseqs_test = numframes_test / seq_len

# MODEL PARAMETERS 
hidden_size = 100

# OPTIMIZATION PARAMETERS
lr = 1.e1 # learning rate
batch_size = 1
save_epoch = 100

max_decay = 20
epsl = 0.

# IO PARAMTERES
models_path = 'models/'
pred_path = 'prediction/'
#features_path = 'features/'
vis_path = 'visualization/'

