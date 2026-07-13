from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class ForecastModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        """Map x [B,T_in,N,F] to predictions [B,T_out,N]."""

    def freeze_temporal(self) -> None:
        """Freeze transferable temporal feature extractor; subclasses define membership."""
        for parameter in self.temporal_parameters():
            parameter.requires_grad = False

    def temporal_parameters(self):
        return iter(())

