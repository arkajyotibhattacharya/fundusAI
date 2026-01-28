"""
Training loop — matches the original starter notebook.
"""

import time

from .config import EPOCHS


def train_model(model, train_generator, val_generator, epochs=None):
    """
    Train a model and return (history, running_time).
    """
    if epochs is None:
        epochs = EPOCHS

    start_time = time.time()

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1,
    )

    running_time = time.time() - start_time
    print(f"Training completed in {running_time:.1f}s")

    return history, running_time
