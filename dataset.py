
"""
dataset.py

Custom PyTorch Dataset for
stock price forecasting.
"""

# =====================================================
# Import Required Libraries
# =====================================================

import torch
from torch.utils.data import Dataset


# =====================================================
# Custom Dataset
# =====================================================

class StockDataset(Dataset):
    """
    Custom Dataset for stock price forecasting.
    """

    def __init__(self, X, y):
        """
        Parameters
        ----------
        X : numpy.ndarray
            Input sequences.

        y : numpy.ndarray
            Target values.
        """

        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Return one sample and its target.
        """
        return self.X[idx], self.y[idx]
