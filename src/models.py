"""
Model architectures.

The baseline CNN matches the original starter notebook exactly.
Team members: add your own model functions here (or in separate files).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model

from .config import (
    IMG_SIZE, SEED,
    CONV_LAYERS_ONE, CONV_LAYERS_TWO, CONV_KERNEL_SIZE, CONV_ACTIVATION,
    POOL_SIZE, FC_UNITS, FC_ACTIVATION, OUTPUT_ACTIVATION,
    LOSS_FUNCTION, OPTIMIZER,
)


def build_baseline_cnn(num_classes):
    """
    Basic 2-conv-layer CNN from the starter notebook.
    """
    tf.random.set_seed(SEED)

    model = keras.Sequential()

    # Input
    model.add(layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)))

    # Conv block 1
    model.add(layers.Conv2D(CONV_LAYERS_ONE, kernel_size=CONV_KERNEL_SIZE, activation=CONV_ACTIVATION))
    model.add(layers.MaxPooling2D(pool_size=POOL_SIZE))

    # Conv block 2
    model.add(layers.Conv2D(CONV_LAYERS_TWO, kernel_size=CONV_KERNEL_SIZE, activation=CONV_ACTIVATION))
    model.add(layers.MaxPooling2D(pool_size=POOL_SIZE))

    # Classification head
    model.add(layers.Flatten())
    model.add(layers.Dense(units=FC_UNITS, activation=FC_ACTIVATION))
    model.add(layers.Dense(units=num_classes, activation=OUTPUT_ACTIVATION))

    model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=['accuracy'])
    return model


def show_model(model):
    """Print summary and plot architecture diagram."""
    model.summary()
    return plot_model(
        model,
        show_shapes=True,
        show_layer_activations=True,
        show_layer_names=True,
    )
