# DATA SIZE
data_path = '../../data/bouncing_balls/3balls/' 
image_suffix = '.jpeg'
image_shape = (16, 16) # single channel images
numframes_train = 50000
numframes_test = 10000
seq_len = 5

# Data size paramters according to user configurations:
frame_dim = image_shape[0] * image_shape[1] # single channel images
seq_dim = frame_dim * seq_len
numseqs_train = numframes_train / seq_len
numseqs_test = numframes_test / seq_len

# MODEL PARAMETERS 
hidden1_size = 50
hidden2_size = 100

# OPTIMIZATION PARAMETERS
lr = 1.e1 # learning rate
batch_size = 1
save_epoch = 100
backup_epoch = 10000

max_decay = 5
epsl = 0.

# IO PARAMTERES
models_path = 'models/'
gc_path = 'grammar_cell_models/'
pred_path = 'prediction/'
#features_path = 'features/'
vis_path = 'visualization/'

# paramers of the gated autoencoder
numfac_=80
numvel_=40
numvelfac_=40
numacc_=20
numaccfac_=10
numjerk_=10

# Paramters according to user configurations:
seq_len_to_train_ = seq_len
seq_len_to_predict_ = seq_len
pred_num = 50

frame_dim = image_shape[0] * image_shape[1] # single channel images
seq_dim = frame_dim * seq_len
numseqs_train = numframes_train / seq_len
numseqs_test = numframes_test / seq_len
