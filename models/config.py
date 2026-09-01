
"""
config.py

Central configuration file for the
MTN Attention-LSTM forecasting project.
"""


import torch

# =====================================================
# Project Configuration
# =====================================================


# =====================================================
# Data Parameters
# =====================================================

TRAIN_RATIO = 0.70

VALIDATION_RATIO = 0.15

TEST_RATIO = 0.15

# =====================================================
# Model Parameters
# =====================================================

INPUT_SIZE = None      # Determined automatically

HIDDEN_SIZE = 64

NUM_LAYERS = 2

DROPOUT = 0.20

FC_UNITS = 32

OUTPUT_SIZE = 1

# =====================================================
# Training Parameters
# =====================================================

BATCH_SIZE = 32

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5


# =====================================================
# Random Seed
# =====================================================

RANDOM_STATE = 42

# =====================================================
# Sequence Parameters
# =====================================================

SEQUENCE_LENGTH = 30

# =====================================================
# Training Parameters
# =====================================================

EPOCHS = 100

PATIENCE = 10


# =====================================================
# Device
# =====================================================

import torch

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# Project Directories and Data
# =====================================================

DATA_PATH = "data/Final_Dataset.csv"

MODEL_DIR = "saved_models/"

RESULT_DIR = "results/"

# =====================================================
# Visualization Parameters
# =====================================================

FIGSIZE = (12, 5)

DPI = 120

STYLE = "ggplot"

# =====================================================
# Model Checkpoints
# =====================================================

STUDY1_BASELINE_MODEL = "saved_models/study1_baseline.pth"

STUDY1_ATTENTION_MODEL = "saved_models/study1_attention.pth"

STUDY2_BASELINE_MODEL = "saved_models/study2_baseline.pth"

STUDY2_ATTENTION_MODEL = "saved_models/study2_attention.pth"

# =====================================================
# Experiment Results
# =====================================================

STUDY1_BASELINE_RESULTS = "results/study1_baseline.csv"

STUDY1_ATTENTION_RESULTS = "results/study1_attention.csv"

STUDY2_BASELINE_RESULTS = "results/study2_baseline.csv"

STUDY2_ATTENTION_RESULTS = "results/study2_attention.csv"
