from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class MLPNodeClassifier(nn.Module):
    """A feature-only baseline with the same two-layer shape as the GNNs."""

    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        x = F.dropout(F.relu(self.linear1(x)), self.dropout, self.training)
        return self.linear2(x)


class TwoLayerGNN(nn.Module):
    def __init__(
        self,
        convolution: type[nn.Module],
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1 = convolution(in_channels, hidden_channels)
        self.conv2 = convolution(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(F.relu(self.conv1(x, edge_index)), self.dropout, self.training)
        return self.conv2(x, edge_index)


class GCN(TwoLayerGNN):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5
    ) -> None:
        super().__init__(GCNConv, in_channels, hidden_channels, out_channels, dropout)


class GraphSAGE(TwoLayerGNN):
    def __init__(
        self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5
    ) -> None:
        super().__init__(SAGEConv, in_channels, hidden_channels, out_channels, dropout)


class GAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
    ) -> None:
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        channels_per_head = hidden_channels // heads
        self.conv1 = GATConv(
            in_channels, channels_per_head, heads=heads, dropout=dropout
        )
        self.conv2 = GATConv(
            hidden_channels, out_channels, heads=1, concat=False, dropout=dropout
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, self.dropout, self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, self.dropout, self.training)
        return self.conv2(x, edge_index)
