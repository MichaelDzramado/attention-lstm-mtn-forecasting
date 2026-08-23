
"""
metrics.py

Utility functions for evaluating
deep learning forecasting models.
"""

# =====================================================
# Import Required Libraries
# =====================================================

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =====================================================
# Evaluation Function
# =====================================================

def evaluate_predictions(y_true, y_pred):
    """
    Compute regression evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Actual values.

    y_pred : array-like
        Predicted values.

    Returns
    -------
    dict
        Dictionary containing regression metrics.
    """

    # =====================================================
    # Convert Inputs to NumPy Arrays
    # =====================================================

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # =====================================================
    # Evaluation Metrics
    # =====================================================

    # Mean Absolute Error
    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    # Mean Squared Error
    mse = mean_squared_error(
        y_true,
        y_pred
    )

    # Root Mean Squared Error
    rmse = np.sqrt(mse)

    # Mean Absolute Percentage Error
    epsilon = 1e-8

    mape = np.mean(
        np.abs(
            (y_true - y_pred) /
            (y_true + epsilon)
        )
    ) * 100

    # Coefficient of Determination
    r2 = r2_score(
        y_true,
        y_pred
    )

    return {

        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2

    }