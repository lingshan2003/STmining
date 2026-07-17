from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GINConv, GPSConv, global_mean_pool


def _local_mlp(channels: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(channels, channels), nn.ReLU(), nn.Linear(channels, channels))


class UniversalGraphEncoder(nn.Module):
    """GPS teaching encoder over a fixed structural feature schema."""

    def __init__(
        self, in_channels: int, hidden_channels: int, layers: int = 3, heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_channels % heads:
            raise ValueError("hidden_channels must be divisible by heads")
        self.input_projection = nn.Linear(in_channels, hidden_channels)
        self.blocks = nn.ModuleList(
            GPSConv(
                hidden_channels,
                GINConv(_local_mlp(hidden_channels), train_eps=True),
                heads=heads,
                dropout=dropout,
            )
            for _ in range(layers)
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        x = self.input_projection(x)
        for block in self.blocks:
            x = block(x, edge_index, batch=batch)
        return global_mean_pool(x, batch)


class GraphTransferClassifier(nn.Module):
    """Task head with an optional learnable feature-space graph prompt."""

    def __init__(
        self,
        encoder: UniversalGraphEncoder,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        use_prompt: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.prompt = nn.Parameter(torch.zeros(in_channels)) if use_prompt else None
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        if self.prompt is not None:
            x = x + self.prompt
        return self.classifier(self.encoder(x, edge_index, batch))

    def freeze_encoder(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
