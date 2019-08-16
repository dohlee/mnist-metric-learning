import torch

# Data.
DIM_IN = 28 * 28
DIM_FEATURE = 10
NUM_CLASSES = 10

# Training.
NUM_TRAIN_DATA = 54000
NUM_VAL_DATA = 6000

# Logging.
LOG_LOSS_EVERY = 100

# Constant string representations.
FC = 'FullyConnected'
ARCFACE = 'ArcFace'
COSFACE = 'CosFace'

SIMPLEDNN = 'SimpleDNN'

# Whether to use GPU or not.
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')