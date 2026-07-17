from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv


class GraphEncoder(nn.Module):
    """A small reusable node encoder shared by pretraining and downstream tasks."""

    def __init__(self, in_channels: int, hidden_channels: int, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        channels = [in_channels, *([hidden_channels] * layers)]
        self.convolutions = nn.ModuleList(
            GCNConv(channels[index], channels[index + 1]) for index in range(layers)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for convolution in self.convolutions:
            x = F.relu(convolution(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)
        return x


class GraphMAE(nn.Module):
    """Masked node-feature autoencoder with a learned mask token."""

    def __init__(self, in_channels: int, hidden_channels: int, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(in_channels))
        self.encoder = GraphEncoder(in_channels, hidden_channels, layers, dropout)
        self.decoder = nn.Linear(hidden_channels, in_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        corrupted = x.clone()
        corrupted[mask] = self.mask_token
        representation = self.encoder(corrupted, edge_index)
        return self.decoder(representation), representation


class PretrainedNodeClassifier(nn.Module):
    def __init__(self, encoder: GraphEncoder, hidden_channels: int, out_channels: int):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x, edge_index))


def symmetric_contrastive_loss(
    first: torch.Tensor, second: torch.Tensor, temperature: float = 0.2
) -> torch.Tensor:
    """Full in-batch node-level InfoNCE in both view directions."""
    if first.shape != second.shape:
        raise ValueError("both contrastive views must have the same shape")
    first = F.normalize(first, dim=-1)
    second = F.normalize(second, dim=-1)
    logits = first @ second.T / temperature
    labels = torch.arange(first.shape[0], device=first.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
