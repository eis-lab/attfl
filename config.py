import os

OPTIMIZER        = 'adam' # 'sgd'
BATCH_SIZE       = 32
LEARNING_RATE    = 0.001
L2WEIGHT_DECAY   = 0.0001

# Bi-LSTM
TIME_STEP_MNIST  = 28
INPUT_SIZE_MNIST = 28
HIDDEN_UNIT      = 64

TIME_STEP_CIFAR  = 64
INPUT_SIZE_CIFAR = 48

# FL
PARTICIPANTS     = 5
NUM_OF_CLIENTS_M = 40
NUM_OF_CLIENTS_C = 40
LOCAL_BATCH_SIZE = BATCH_SIZE # Adjustable 
LOCAL_EPOCH      = 5
SHARDS           = 5
ROUNDS           = 50

# dataset dir
DATASET_DIR      = './dataset/'

# model
CODE_VERSION     = '1'
FED_MODEL        = 'AnonymousFed'
MODEL_SAVE_PATH  = './saved_model/'

# tensorboard
TB_DIR           = './tensorboard/'

# log
PRINT_LOG_INTERVAL = 1 

# gpu
GPU_NUM = 0