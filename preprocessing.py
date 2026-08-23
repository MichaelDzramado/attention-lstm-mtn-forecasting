
"""
preprocessing.py

Utility functions for preparing stock market data for
deep learning models.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the engineered dataset.

    Parameters
    ----------
    filepath : str
        Path to the dataset.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(filepath)

    # Convert date column to datetime
    df["Daily Date"] = pd.to_datetime(df["Daily Date"])

    # Ensure chronological order
    df = df.sort_values("Daily Date").reset_index(drop=True)

    return df


def select_features(df: pd.DataFrame,
                    target_column: str = "Target_Close"):
    """
    Separate predictors and forecasting target.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    target_column : str
        Forecast target.

    Returns
    -------
    X : pd.DataFrame
        Predictor variables.

    y : pd.Series
        Target variable.
    """

    X = df.drop(columns=[
        target_column,
        "Daily Date",
        "Share Code",
        "Month_Name"
    ])

    y = df[target_column]

    return X, y


def split_dataset(X,
                  y,
                  train_size=0.70,
                  val_size=0.15):
    """
    Perform chronological train-validation-test split.

    Returns
    -------
    X_train, X_val, X_test,
    y_train, y_val, y_test
    """

    n = len(X)

    train_end = int(train_size * n)

    val_end = train_end + int(val_size * n)

    X_train = X.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]

    y_train = y.iloc[:train_end]
    y_val = y.iloc[train_end:val_end]
    y_test = y.iloc[val_end:]

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )



def scale_features(
    X_train,
    X_val,
    X_test,
    y_train=None,
    y_val=None,
    y_test=None,
    target_scaling=False
):
    """
    Scale predictor variables using MinMaxScaler.

    Optionally scales the target variable for
    comparative deep learning experiments.

    Parameters
    ----------
    X_train, X_val, X_test : pandas.DataFrame or ndarray
        Predictor datasets.

    y_train, y_val, y_test : pandas.Series or ndarray, optional
        Target datasets.

    target_scaling : bool, default=False
        Whether to scale the forecasting target.

    Returns
    -------
    X_train_scaled
    X_val_scaled
    X_test_scaled

    y_train_processed
    y_val_processed
    y_test_processed

    feature_scaler

    target_scaler (None if target_scaling=False)
    """

    # =====================================================
    # Scale Predictor Variables
    # =====================================================

    feature_scaler = MinMaxScaler()

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    # =====================================================
    # Process Target Variable
    # =====================================================

    if target_scaling:

        target_scaler = MinMaxScaler()

        # Convert pandas Series to NumPy arrays if necessary
        if hasattr(y_train, "values"):
            y_train = y_train.values
            y_val = y_val.values
            y_test = y_test.values

        y_train_processed = target_scaler.fit_transform(
            y_train.reshape(-1, 1)
        )

        y_val_processed = target_scaler.transform(
            y_val.reshape(-1, 1)
        )

        y_test_processed = target_scaler.transform(
            y_test.reshape(-1, 1)
        )

    else:

        target_scaler = None

        y_train_processed = y_train
        y_val_processed = y_val
        y_test_processed = y_test

    # =====================================================
    # Return Processed Data
    # =====================================================

    return (

        X_train_scaled,
        X_val_scaled,
        X_test_scaled,

        y_train_processed,
        y_val_processed,
        y_test_processed,

        feature_scaler,
        target_scaler

    )



def create_sequences(X, y, sequence_length):
    """
    Convert tabular data into sequential data
    for LSTM training.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix.

    y : array-like
        Target values.

    sequence_length : int
        Number of previous observations.

    Returns
    -------
    X_seq : numpy.ndarray
        Sequential feature data.

    y_seq : numpy.ndarray
        Corresponding targets.
    """

    import numpy as np

    X_seq = []
    y_seq = []

    for i in range(len(X) - sequence_length):

        X_seq.append(
            X[i:i + sequence_length]
        )

        y_seq.append(
            y.iloc[i + sequence_length]
            if hasattr(y, "iloc")
            else y[i + sequence_length]
        )

    X_seq = np.asarray(X_seq, dtype=np.float32)

    y_seq = np.asarray(
        y_seq,
        dtype=np.float32
    ).reshape(-1, 1)

    return X_seq, y_seq