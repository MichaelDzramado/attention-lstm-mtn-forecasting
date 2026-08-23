
"""
attention_lstm.py

Attention-Enhanced Long Short-Term Memory (Attention-LSTM)
for one-step-ahead stock price forecasting.

Author:
Deep Learning Semester Project
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

from models.attention import Attention


# =====================================================
# Attention-Enhanced LSTM Model
# =====================================================

class AttentionLSTM(nn.Module):
    """
    Attention-Enhanced Long Short-Term Memory Network.

    The model first extracts temporal features using
    stacked LSTM layers before learning which historical
    observations are most important through a custom
    attention mechanism.
    """

    def __init__(self, input_size):
        """
        Initialize the proposed Attention-LSTM model.

        Parameters
        ----------
        input_size : int
            Number of predictor variables.
        """

        super().__init__()

        # =================================================
        # LSTM Feature Extractor
        # =================================================

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_first=True
        )

        # =================================================
        # Attention Layer
        # =================================================

        self.attention = Attention(HIDDEN_SIZE)

        # =================================================
        # Normalization Layer
        # =================================================

        self.layer_norm = nn.LayerNorm(HIDDEN_SIZE)

        # =================================================
        # Regularization
        # =================================================

        self.dropout = nn.Dropout(DROPOUT)

        # =================================================
        # Prediction Head
        # =================================================

        self.fc1 = nn.Linear(
           HIDDEN_SIZE,
           FC_UNITS
        )

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(
            FC_UNITS,
            OUTPUT_SIZE
        )
        
        # =================================================
        # Initialize Weights
        # =================================================

        self._initialize_weights()

    # =====================================================
    # Weight Initialization
    # =====================================================

    def _initialize_weights(self):
        """
        Initialize trainable parameters to improve
        convergence during optimization.
        """

        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

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
        Perform forward propagation.

        Parameters
        ----------
        x : torch.Tensor

            Shape:
            (batch_size,
             sequence_length,
             input_size)

        Returns
        -------
        prediction : torch.Tensor

        attention_weights : torch.Tensor
        """

        # ---------------------------------------------
        # Learn temporal representations
        # ---------------------------------------------

        lstm_outputs, _ = self.lstm(x)

        # ---------------------------------------------
        # Learn important historical observations
        # ---------------------------------------------

        context_vector, attention_weights = self.attention(
            lstm_outputs
        )

        # ---------------------------------------------
        # Stabilize feature distribution
        # ---------------------------------------------

        x = self.layer_norm(context_vector)

        # ---------------------------------------------
        # Regularization
        # ---------------------------------------------

        x = self.dropout(x)

        # ---------------------------------------------
        # Fully connected Layer
        # ---------------------------------------------

        x = self.fc1(x)

        x = self.relu(x)

        x = self.dropout(x)


        # ---------------------------------------------
        # Output Layer
        # ---------------------------------------------

        prediction = self.fc2(x)

        return prediction, attention_weights
