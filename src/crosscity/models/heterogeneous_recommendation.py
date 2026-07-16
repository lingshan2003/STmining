from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv


class HeterogeneousEmbedding(nn.Module):
    """Type-specific ID embeddings without message passing."""

    def __init__(self, num_nodes: dict[str, int], hidden_channels: int):
        super().__init__()
        self.embedding = nn.ModuleDict({
            node_type: nn.Embedding(count, hidden_channels)
            for node_type, count in num_nodes.items()
        })
        for table in self.embedding.values():
            nn.init.xavier_uniform_(table.weight)

    def encode(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        return {
            node_type: self.embedding[node_type].weight
            for node_type in graph.node_types
        }

    @staticmethod
    def score(
        representations: dict[str, torch.Tensor],
        users: torch.Tensor,
        products: torch.Tensor,
    ) -> torch.Tensor:
        return (representations["user"][users] * representations["product"][products]).sum(-1)


class HeterogeneousGraphSAGE(HeterogeneousEmbedding):
    """Use an independent GraphSAGE transformation for every edge type."""

    def __init__(
        self,
        num_nodes: dict[str, int],
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_channels: int,
        num_layers: int = 2,
    ):
        super().__init__(num_nodes, hidden_channels)
        self.layers = nn.ModuleList([
            HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            }, aggr="sum")
            for _ in range(num_layers)
        ])

    def encode(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        representations = super().encode(graph)
        for layer in self.layers:
            updated = layer(representations, graph.edge_index_dict)
            representations = {
                node_type: (
                    torch.relu(updated[node_type]) + representations[node_type]
                    if node_type in updated else representations[node_type]
                )
                for node_type in representations
            }
        return representations
