from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv


class ScalableMLP(nn.Module):
    """Feature-only node classifier used to audit whether graph sampling helps."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SampledGraphSAGE(nn.Module):
    """GraphSAGE whose forward pass works on either a full graph or sampled block."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        *,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        channels = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        self.convolutions = nn.ModuleList([
            SAGEConv(channels[index], channels[index + 1])
            for index in range(num_layers)
        ])
        self.dropout = dropout

    @property
    def num_layers(self) -> int:
        return len(self.convolutions)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for index, convolution in enumerate(self.convolutions):
            x = convolution(x, edge_index)
            if index != self.num_layers - 1:
                x = torch.relu(x)
                x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return x

    @torch.no_grad()
    def layerwise_inference(self, x_all, loader, device: torch.device | str) -> torch.Tensor:
        """Exact one-layer-at-a-time inference without placing all edges on the GPU.

        ``loader`` must enumerate every node in ID order, use one-hop ``[-1]``
        sampling, and have ``shuffle=False``. PyG stores seed nodes first in every
        sampled batch, so only the first ``batch_size`` outputs are collected.
        """
        self.eval()
        for layer_index, convolution in enumerate(self.convolutions):
            outputs = []
            for batch in loader:
                node_ids = batch.n_id
                features = x_all[node_ids].to(device)
                edge_index = batch.edge_index.to(device)
                output = convolution(features, edge_index)[: batch.batch_size]
                if layer_index != self.num_layers - 1:
                    output = torch.relu(output)
                outputs.append(output.cpu())
            x_all = torch.cat(outputs, dim=0)
        return x_all
