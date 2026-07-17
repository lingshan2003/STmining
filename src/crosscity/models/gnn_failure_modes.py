from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, degree


class DeepGCN(nn.Module):
    """A depth-controlled GCN with optional residual paths and per-node LayerNorm."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        *,
        residual: bool = False,
        normalization: str = "none",
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if normalization not in {"none", "layer"}:
            raise ValueError("normalization must be 'none' or 'layer'")
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.convolutions = nn.ModuleList(
            GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers)
        )
        self.normalizations = nn.ModuleList(
            nn.LayerNorm(hidden_channels) if normalization == "layer" else nn.Identity()
            for _ in range(num_layers)
        )
        self.classifier = nn.Linear(hidden_channels, out_channels)
        self.residual = residual
        self.dropout = dropout

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.encoder(x))
        for convolution, normalization in zip(self.convolutions, self.normalizations):
            update = normalization(convolution(h, edge_index))
            h = h + F.relu(update) if self.residual else F.relu(update)
            h = F.dropout(h, self.dropout, training=self.training)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(x, edge_index))


def normalized_propagation(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """One parameter-free GCN propagation step, including self loops."""
    edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.shape[0])
    source, target = edge_index
    target_degree = degree(target, x.shape[0], dtype=x.dtype)
    weights = target_degree[source].clamp_min(1).pow(-0.5)
    weights = weights * target_degree[target].clamp_min(1).pow(-0.5)
    output = torch.zeros_like(x)
    output.index_add_(0, target, x[source] * weights.unsqueeze(-1))
    return output


@torch.no_grad()
def propagation_trace(
    x: torch.Tensor, edge_index: torch.Tensor, steps: int = 16
) -> list[dict[str, float]]:
    """Track variance, adjacent Dirichlet energy, and random-pair cosine by depth."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    h = x.float()
    source, target = edge_index
    generator = torch.Generator(device="cpu").manual_seed(0)
    pair_count = min(20_000, max(h.shape[0], 1) * 4)
    left = torch.randint(h.shape[0], (pair_count,), generator=generator).to(h.device)
    right = torch.randint(h.shape[0], (pair_count,), generator=generator).to(h.device)
    records = []
    for step in range(steps + 1):
        normalized = F.normalize(h, p=2, dim=-1)
        cosine = (normalized[left] * normalized[right]).sum(-1).mean()
        energy = (h[source] - h[target]).square().sum(-1).mean()
        records.append(
            {
                "step": float(step),
                "feature_variance": float(h.var(dim=0, unbiased=False).mean()),
                "dirichlet_energy": float(energy),
                "mean_pairwise_cosine": float(cosine),
            }
        )
        h = normalized_propagation(h, edge_index)
    return records
