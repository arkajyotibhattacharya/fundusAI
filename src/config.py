"""
Shared configuration — change values here and they propagate everywhere.
"""

SEED = 1
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
TARGET_COL = 'labels'

# Split ratios
TEST_SIZE = 0.2   # fraction of all data for test
VAL_SIZE = 0.2    # fraction of training data for validation

# Baseline CNN hyperparameters
CONV_LAYERS_ONE = 32
CONV_LAYERS_TWO = 64
CONV_KERNEL_SIZE = (3, 3)
CONV_ACTIVATION = 'relu'
POOL_SIZE = (2, 2)
FC_UNITS = 128
FC_ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'softmax'
LOSS_FUNCTION = 'categorical_crossentropy'
OPTIMIZER = 'Adam'
EPOCHS = 1  # increase when ready

# Plot colors (from original notebook)
COLORS_DARK = ["#1F1F1F", "#313131", '#636363', '#AEAEAE', '#DADADA']
COLORS_RED = ["#331313", "#582626", '#9E1717', '#D35151', '#E9B4B4']
COLORS_GREEN = ['#01411C', '#4B6F44', '#4F7942', '#74C365', '#D0F0C0']
