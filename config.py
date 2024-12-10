import json
import os
from pathlib import Path



#Params to tune:
NUM_MP_STEPS = 10  # Message passing steps
INPUT_SEQUENCE_LENGTH = 6  # Number of input positions (5 velocities)

# Dataset name
DATASET_NAME = "Water100"

# Data paths
BASE_DIR = Path("/data/rye/5835/utils_final")  # Adjust as needed
DATA_DIR = Path("/data/rye/5835/datasets") / f"{DATASET_NAME}_hdf5"
METADATA_PATH = DATA_DIR / "metadata.json"
OUT_DIR = Path("/data/rye/5835/models") / f"{DATASET_NAME}_mp{NUM_MP_STEPS}_sl{INPUT_SEQUENCE_LENGTH}"






# Load metadata
with open(METADATA_PATH, 'r') as f:
    METADATA = json.load(f)

# Extract constants from metadata
BOUNDS = METADATA['bounds']
SEQUENCE_LENGTH = METADATA['sequence_length']
CONNECTIVITY_RADIUS = METADATA['default_connectivity_radius']
DIM = METADATA['dim']
DT = METADATA['dt']

# Normalization constants from metadata
VELOCITY_MEAN = METADATA['vel_mean']
VELOCITY_STD = METADATA['vel_std']
ACCELERATION_MEAN = METADATA['acc_mean']
ACCELERATION_STD = METADATA['acc_std']

# Model hyperparameters (un-optimized)
NOISE_STD = 3e-4
LATENT_DIM = 128
NUM_LAYERS = 3


# Node features
PARTICLE_TYPE_EMBEDDING_DIM = 9
NUM_PARTICLE_TYPES = 9  # water, sand, goop, rigid, boundary etc.

# Training parameters
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
LEARNING_RATE_DECAY = 0.1
MAX_EPOCHS = 50
NUM_WORKERS = 4

# Device
DEVICE = 'cuda'  # or 'cpu'

# Splits
TRAIN_SPLIT = 'train'
VALID_SPLIT = 'valid'
TEST_SPLIT = 'test'

# Feature dimensions
NODE_IN_DIM = (INPUT_SEQUENCE_LENGTH-1) * DIM + PARTICLE_TYPE_EMBEDDING_DIM  # 5 velocities + particle type
EDGE_IN_DIM = DIM  # relative displacement

# Random seed
SEED = 42