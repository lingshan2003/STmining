from __future__ import annotations

import torch
from torch import nn

from .base import ForecastModel


class SharedLSTM(ForecastModel):
    """One temporal model shared by every sensor, hence independent of graph size."""

    def __init__(self, output_steps: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.output_steps = output_steps
        self.temporal_encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, output_steps)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        batch, steps, nodes, features = x.shape
        sequences = x.permute(0, 2, 1, 3).reshape(batch * nodes, steps, features)
        _, (hidden, _) = self.temporal_encoder(sequences)
        prediction = self.head(hidden[-1]).reshape(batch, nodes, self.output_steps)
        return prediction.transpose(1, 2)

    def temporal_parameters(self):
        return self.temporal_encoder.parameters()

