from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn.models import LightGCN


class MatrixFactorization(nn.Module):
    """User/item embeddings without graph propagation."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 64):
        super().__init__()
        self.num_users = num_users
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

    def get_embedding(self, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        del edge_index
        return torch.cat((self.user_embedding.weight, self.item_embedding.weight), dim=0)


def build_lightgcn(
    num_users: int,
    num_items: int,
    embedding_dim: int = 64,
    num_layers: int = 3,
    alpha: float | torch.Tensor | None = None,
) -> LightGCN:
    """Build LightGCN with optional weights for ego and propagated layers."""
    return LightGCN(num_users + num_items, embedding_dim, num_layers, alpha=alpha)
