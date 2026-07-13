from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import ForecastModel


class HistoricalAverage:
    """Average by time-of-week and sensor, with a sensor mean fallback."""

    def __init__(self, steps_per_day: int = 288) -> None:
        self.period = steps_per_day * 7
        self.profile: np.ndarray | None = None
        self.fallback: np.ndarray | None = None

    def fit(self, values: np.ndarray, mask: np.ndarray) -> "HistoricalAverage":
        sums = np.zeros((self.period, values.shape[1]), dtype=np.float64)
        counts = np.zeros_like(sums)
        slots = np.arange(len(values)) % self.period
        np.add.at(sums, slots, np.where(mask, values, 0))
        np.add.at(counts, slots, mask)
        node_counts = mask.sum(axis=0)
        self.fallback = np.divide(
            np.where(mask, values, 0).sum(axis=0), node_counts,
            out=np.zeros(values.shape[1]), where=node_counts > 0,
        )
        self.profile = np.divide(sums, counts, out=np.broadcast_to(self.fallback, sums.shape).copy(), where=counts > 0)
        return self

    def predict(self, starts: np.ndarray, output_steps: int) -> np.ndarray:
        if self.profile is None:
            raise RuntimeError("fit HistoricalAverage before predict")
        slots = (starts[:, None] + np.arange(output_steps)[None, :]) % self.period
        return self.profile[slots]


class SharedMLP(ForecastModel):
    def __init__(self, input_steps: int, output_steps: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_steps, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_steps)
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        # [B,T,N,1] -> [B,N,T] -> node-shared MLP -> [B,T_out,N]
        return self.network(x[..., 0].transpose(1, 2)).transpose(1, 2)

    def temporal_parameters(self):
        return self.network[0].parameters()

