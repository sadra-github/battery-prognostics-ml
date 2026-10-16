"""Baseline model components. Validation and training remain experiment-specific."""

import torch
from torch import nn


class LSTMRULPredictor(nn.Module):
    """LSTM regressor for windows shaped (batch, sequence, features).

    The model outputs one scalar RUL estimate per input window. It does not
    define the target or data split; those must be set in the experiment.
    """
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError("input_size, hidden_size, and num_layers must be positive")
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input shape (batch, sequence, features)")
        sequence_output, _ = self.lstm(x)
        return self.regressor(sequence_output[:, -1, :]).squeeze(-1)


class PINNSOHEstimator(nn.Module):
    """Neural SOH regressor with an optional externally defined physics loss.

    This class does not implement a universal battery-physics constraint.
    Supply a problem-specific constraint function before describing a training
    procedure as physics-informed.
    """
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        if input_size < 1 or hidden_size < 1:
            raise ValueError("input_size and hidden_size must be positive")
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)

    @staticmethod
    def physics_loss(prediction: torch.Tensor, constraint_fn=None) -> torch.Tensor:
        """Evaluate a user-supplied physical constraint or raise if absent."""
        if constraint_fn is None:
            raise NotImplementedError(
                "Define a dataset- and physics-specific constraint_fn before use."
            )
        return constraint_fn(prediction)
