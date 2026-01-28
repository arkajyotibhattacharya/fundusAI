"""
Model evaluation — plots and metrics from the original starter notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from warnings import filterwarnings

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from .config import COLORS_DARK, COLORS_RED, COLORS_GREEN


def plot_training_history(history):
    """Plot train/val accuracy and loss over epochs (original notebook style)."""
    filterwarnings('ignore')

    epochs_list = list(range(len(history.history['accuracy'])))

    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.text(
        s='Epochs vs. Training and Validation Accuracy/Loss',
        size=18, fontweight='bold', fontname='monospace',
        color=COLORS_DARK[1], y=1, x=0.28, alpha=0.8,
    )

    sns.despine()
    ax[0].plot(epochs_list, train_acc, marker='o',
               markerfacecolor=COLORS_GREEN[2], color=COLORS_GREEN[3],
               label='Training Accuracy')
    ax[0].plot(epochs_list, val_acc, marker='o',
               markerfacecolor=COLORS_RED[2], color=COLORS_RED[3],
               label='Validation Accuracy')
    ax[0].legend(frameon=False)
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')

    sns.despine()
    ax[1].plot(epochs_list, train_loss, marker='o',
               markerfacecolor=COLORS_GREEN[2], color=COLORS_GREEN[3],
               label='Training Loss')
    ax[1].plot(epochs_list, val_loss, marker='o',
               markerfacecolor=COLORS_RED[2], color=COLORS_RED[3],
               label='Validation Loss')
    ax[1].legend(frameon=False)
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Training & Validation Loss')

    fig.show()


def evaluate_model(model, test_generator):
    """Predict on test set and print accuracy, confusion matrix, and classification report."""
    pred_probs = model.predict(test_generator)
    pred_classes = np.argmax(pred_probs, axis=1)
    true_classes = test_generator.classes
    class_names = list(test_generator.class_indices.keys())

    acc = accuracy_score(true_classes, pred_classes)
    print("Test accuracy:", acc)

    cm = confusion_matrix(true_classes, pred_classes)
    print("Confusion Matrix:\n", cm)

    report = classification_report(true_classes, pred_classes, target_names=class_names)
    print("Classification Report:\n", report)

    return pred_probs, pred_classes, true_classes
