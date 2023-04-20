import torch

# Hyperparameters
H, W = 256, 256
BATCH_SIZE = 32
VISUALIZATION_BATCH_SIZE = int(5*5)
LEARNING_RATE = 0.002
GRAD_ACCUMULATE_FACTOR = 8  # minibatch_size = BATCH_SIZE / GRAD_ACCUMULATE_FACTOR
G_LAZY_REG_FACTOR = 4 # apply once every 4 minibatch
D_LAZY_REG_FACTOR = 8 # apply once every 16 minibatch
N_CRITICS = 1

# Generator and Discriminator
NUM_FEATURE_MAP = {512:64, 256: 64, 128:128, 64:256, 32:512, 16:512, 8:512, 4:512, 2:512}
NUM_MAPPING_LAYER = 8
LATENT_SIZE = 512

# Gradient Penalty weight
GP_LAMBDA = 10
R1_GAMMA = 10
PL_WEIGHT = 2
PL_BATCH_SIZE_RATIO = 0.5
PL_DECAY = 0.01

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')