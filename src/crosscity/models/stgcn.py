from __future__ import annotations

import torch
from torch import nn

from .base import ForecastModel


class TemporalGatedConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, 2 * out_channels, (kernel_size, 1), padding=(padding, 0))
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values, gates = self.conv(x).chunk(2, dim=1)
        return (values + self.residual(x)) * torch.sigmoid(gates)


class GraphConv(nn.Module):
    """GCN weight is channel-specific, never node-specific, so it transfers across cities."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.linear = nn.Linear(channels, channels)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        # x [B,C,T,N]; aggregate neighbors at N then mix channels.
        aggregated = torch.einsum("nm,bctm->bctn", adjacency, x)
        return torch.relu(self.linear(aggregated.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))


class STBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.temporal_in = TemporalGatedConv(in_channels, hidden_dim)
        self.graph = GraphConv(hidden_dim)
        self.temporal_out = TemporalGatedConv(hidden_dim, hidden_dim)
        self.norm = nn.BatchNorm2d(hidden_dim)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return self.norm(self.temporal_out(self.graph(self.temporal_in(x), adjacency)))


class STGCN(ForecastModel):
    def __init__(self, input_steps: int, output_steps: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.block1 = STBlock(1, hidden_dim)
        self.block2 = STBlock(hidden_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, (input_steps, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, output_steps, 1),
        )

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor | None = None) -> torch.Tensor:
        if adjacency is None:
            raise ValueError("STGCN requires an adjacency matrix")
        features = x.permute(0, 3, 1, 2)  # [B,F,T,N]
        features = self.block2(self.block1(features, adjacency), adjacency)
        return self.head(features).squeeze(2)  # [B,T_out,N]

    def temporal_parameters(self):
        modules = [self.block1.temporal_in, self.block1.temporal_out,
                   self.block2.temporal_in, self.block2.temporal_out]
        for module in modules:
            yield from module.parameters()

