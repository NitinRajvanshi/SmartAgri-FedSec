import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "iot_data_ciciot2023.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Federated learning settings
NUM_CLIENTS = 2
NUM_ROUNDS = 2
LOCAL_EPOCHS = 1
BATCH_SIZE = 64
LR = 1e-3

# Homomorphic encryption toggle
USE_HE =True # set to run FL without encryption

# Train/val split
TEST_SIZE = 0.2
RANDOM_STATE = 42
