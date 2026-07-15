from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GAE, VGAE, GCNConv


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv2(F.relu(self.conv1(x, edge_index)), edge_index)


class VariationalGCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_channels: int):
        super().__init__()
        self.shared = GCNConv(in_channels, hidden_channels)
        self.mean = GCNConv(hidden_channels, latent_channels)
        self.log_standard_deviation = GCNConv(hidden_channels, latent_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.relu(self.shared(x, edge_index))
        return self.mean(hidden, edge_index), self.log_standard_deviation(hidden, edge_index)


def build_gae(in_channels: int, hidden_channels: int = 64, latent_channels: int = 32) -> GAE:
    return GAE(GCNEncoder(in_channels, hidden_channels, latent_channels))


def build_vgae(in_channels: int, hidden_channels: int = 64, latent_channels: int = 32) -> VGAE:
    return VGAE(VariationalGCNEncoder(in_channels, hidden_channels, latent_channels))
