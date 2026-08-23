
"""
lstm.py

Baseline Long Short-Term Memory (LSTM) model for
one-step-ahead stock price forecasting.

Author:
Michael and Agbesi
"""

# =====================================================
# Import Required Libraries
# =====================================================

import torch
import torch.nn as nn

from config import (
    HIDDEN_SIZE,
    NUM_LAYERS,
    DROPOUT,
    FC_UNITS,
    OUTPUT_SIZE
)


# =====================================================
# Baseline LSTM Model
# =====================================================

class StockLSTM(nn.Module):
    """
    Baseline Long Short-Term Memory (LSTM) model.

    The network learns temporal relationships from
    historical stock market observations and predicts
    the next trading day's closing price.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        dropout,
        fc_units,
        output_size
    ):
        """
        Initialize the baseline LSTM model.

        Parameters
        ----------
        input_size : int
            Number of input features.
        """

        super().__init__()

        # -------------------------------------------------
        # Stacked LSTM Layers
        # -------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # -------------------------------------------------
        # Regularization Layer
        # -------------------------------------------------

        self.dropout = nn.Dropout(dropout)

        # -------------------------------------------------
        # Fully Connected Layers
        # -------------------------------------------------

        self.fc1 = nn.Linear(hidden_size, fc_units)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(fc_units, output_size)

        # Initialize network weights
        self._initialize_weights()

    # =====================================================
    # Weight Initialization
    # =====================================================

    def _initialize_weights(self):
        """
        Initialize network weights to improve convergence.
        """

        # Xavier initialization for Linear layers
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        # Orthogonal initialization for recurrent weights
        for name, param in self.lstm.named_parameters():

            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

            elif "weight_hh" in name:
                nn.init.orthogonal_(param)

            elif "bias" in name:
                nn.init.zeros_(param)

    # =====================================================
    # Forward Propagation
    # =====================================================

    def forward(self, x):
        """
        Forward propagation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape
            (batch_size, sequence_length, input_size)

        Returns
        -------
        torch.Tensor
            Predicted closing price.
        """

        # Process sequential data through the LSTM
        output, (hidden_state, cell_state) = self.lstm(x)

        # Extract the hidden representation from
        # the final time step
        x = output[:, -1, :]

        # Apply dropout for regularization
        x = self.dropout(x)

        # Fully connected layers
        x = self.fc1(x)

        x = self.relu(x)

        prediction = self.fc2(x)

        return prediction
