
"""
visualization.py

Utility functions for visualizing
deep learning model performance.
"""

# =====================================================
# Import Libraries
# =====================================================

import matplotlib.pyplot as plt
import numpy as np

from config import (
    FIGSIZE,
    DPI,
    STYLE
)
# =====================================================
# Global Plot Style
# =====================================================

plt.style.use("ggplot")

FIGSIZE = (10,6)

# =====================================================
# Training Loss
# =====================================================

def plot_loss(history):
    """
    Plot training and validation loss.
    """

    plt.figure(
        figsize=FIGSIZE,
        dpi=DPI
    )

    plt.plot(
        history.train_loss,
        label="Training Loss",
        linewidth=2
    )

    plt.plot(
        history.val_loss,
        label="Validation Loss",
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        "Training and Validation Loss"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# =====================================================
# Learning Rate
# =====================================================

def plot_learning_rate(history):
    """
    Plot learning rate schedule.
    """

    plt.figure(
        figsize=FIGSIZE,
        dpi=DPI
    )

    plt.plot(
        history.learning_rate,
        linewidth=2
    )

    plt.xlabel("Epoch")

    plt.ylabel("Learning Rate")

    plt.title(
        "Learning Rate Schedule"
    )

    plt.tight_layout()

    plt.show()

# =====================================================
# Actual vs Predicted Prices
# =====================================================

def plot_predictions(y_true, y_pred):
    """
    Plot actual and predicted stock prices.

    Parameters
    ----------
    y_true : array-like
        Actual stock prices.

    y_pred : array-like
        Predicted stock prices.
    """

    plt.figure(
        figsize=FIGSIZE,
        dpi=DPI
    )

    plt.plot(
        y_true,
        label="Actual",
        linewidth=2
    )

    plt.plot(
        y_pred,
        label="Predicted",
        linewidth=2
    )

    plt.xlabel("Observation")

    plt.ylabel("Closing Price (GH¢)")

    plt.title("Actual vs Predicted Closing Prices")

    plt.legend()

    plt.tight_layout()

    plt.show()

# =====================================================
# Residual Plot
# =====================================================


def plot_residuals(y_true, y_pred):
    """
    Plot prediction residuals.
    """
    import numpy as np

    # Ensure both arrays are one-dimensional
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    residuals = y_true - y_pred

    plt.figure(
        figsize=FIGSIZE,
        dpi=DPI
    )

    plt.scatter(
        y_pred,
        residuals,
        alpha=0.6
    )

    plt.axhline(
        y=0,
        color="red",
        linestyle="--"
    )

    plt.xlabel("Predicted Price")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.tight_layout()

    plt.show()

    # =====================================================
# Attention Weights
# =====================================================

def plot_attention_weights(attention):
    """
    Plot attention weights for one sequence.

    Parameters
    ----------
    attention : numpy.ndarray
        Attention weights for one sample.
    """

    plt.figure(
        figsize=FIGSIZE,
        dpi=DPI
    )

    plt.imshow(
        attention.reshape(1, -1),
        aspect="auto",
        cmap="viridis"
    )

    plt.yticks([])

    plt.xlabel("Time Step")

    plt.title("Attention Weight Distribution")

    plt.colorbar()

    plt.tight_layout()

    plt.show()
