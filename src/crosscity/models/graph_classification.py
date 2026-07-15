from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv, GINConv, GPSConv, global_mean_pool


class GraphMLP(nn.Module):
    """Feature-only baseline: encode atoms independently, then mean-pool each graph."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(),
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        del edge_index
        graph_features = global_mean_pool(self.encoder(x), batch)
        return self.classifier(F.dropout(graph_features, self.dropout, self.training))


class GraphMessagePassingClassifier(nn.Module):
    def __init__(self, convolutions: list[nn.Module], hidden_channels: int,
                 out_channels: int, dropout: float = 0.5):
        super().__init__()
        self.convolutions = nn.ModuleList(convolutions)
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        for convolution in self.convolutions:
            x = F.dropout(F.relu(convolution(x, edge_index)), self.dropout, self.training)
        return self.classifier(global_mean_pool(x, batch))


class GraphGCN(GraphMessagePassingClassifier):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int = 3, dropout: float = 0.5):
        channels = [in_channels, *([hidden_channels] * layers)]
        convolutions = [GCNConv(channels[i], channels[i + 1]) for i in range(layers)]
        super().__init__(convolutions, hidden_channels, out_channels, dropout)


def _gin_mlp(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels))


class GraphGIN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.convolutions = nn.ModuleList(
            GINConv(_gin_mlp(hidden_channels), train_eps=True) for _ in range(layers)
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.input_projection(x))
        for convolution in self.convolutions:
            x = F.dropout(F.relu(convolution(x, edge_index)), self.dropout, self.training)
        return self.classifier(global_mean_pool(x, batch))


class GraphGPS(nn.Module):
    """GPS blocks combine local GIN aggregation with global multi-head attention."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 layers: int = 3, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            GPSConv(
                hidden_channels,
                GINConv(_gin_mlp(hidden_channels), train_eps=True),
                heads=heads,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x, edge_index, batch=batch)
        return self.classifier(global_mean_pool(x, batch))
