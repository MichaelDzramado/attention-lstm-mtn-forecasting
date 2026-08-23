
"""
attention.py

Custom attention mechanism for sequential
stock price forecasting.
"""

# =====================================================
# Import Required Libraries
# =====================================================

import torch
import torch.nn as nn


# =====================================================
# Attention Layer
# =====================================================

class Attention(nn.Module):
    """
    Custom attention layer.

    Computes attention weights over the hidden
    states produced by an LSTM.
    """

    def __init__(self, hidden_size):
        """
        Parameters
        ----------
        hidden_size : int
            Dimension of the LSTM hidden state.
        """

        super().__init__()

        # Linear layer used to compute attention scores
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_outputs):
        """
        Forward propagation.

        Parameters
        ----------
        lstm_outputs : torch.Tensor

            Shape:
            (batch_size,
             sequence_length,
             hidden_size)

        Returns
        -------
        context_vector

        attention_weights
        """

        # ---------------------------------------------
        # Compute attention scores
        # ---------------------------------------------

        scores = self.attention(lstm_outputs)

        # ---------------------------------------------
        # Normalize scores
        # ---------------------------------------------

        attention_weights = torch.softmax(
            scores,
            dim=1
        )

        # ---------------------------------------------
        # Weighted combination
        # ---------------------------------------------

        context_vector = torch.sum(
            attention_weights * lstm_outputs,
            dim=1
        )

        return context_vector, attention_weights
