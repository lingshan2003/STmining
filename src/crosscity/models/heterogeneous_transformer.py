from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, HeteroConv, Linear, SAGEConv


Metadata = tuple[list[str], list[tuple[str, str, str]]]


class HeterogeneousInputEncoder(nn.Module):
    """Project type-specific features, or learn IDs for featureless node types."""

    def __init__(
        self,
        input_dims: dict[str, int | None],
        num_nodes: dict[str, int],
        hidden_channels: int,
    ):
        super().__init__()
        self.projections = nn.ModuleDict({
            node_type: Linear(input_dim, hidden_channels)
            for node_type, input_dim in input_dims.items()
            if input_dim is not None
        })
        self.embeddings = nn.ModuleDict({
            node_type: nn.Embedding(num_nodes[node_type], hidden_channels)
            for node_type, input_dim in input_dims.items()
            if input_dim is None
        })
        for embedding in self.embeddings.values():
            nn.init.xavier_uniform_(embedding.weight)

    def forward(self, graph: HeteroData) -> dict[str, torch.Tensor]:
        output = {}
        for node_type in graph.node_types:
            if node_type in self.projections:
                output[node_type] = torch.relu(self.projections[node_type](graph[node_type].x))
            else:
                output[node_type] = self.embeddings[node_type].weight
        return output


class HeterogeneousMLPClassifier(nn.Module):
    """Classify target nodes from their own attributes without graph edges."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, graph: HeteroData) -> torch.Tensor:
        return self.network(graph["author"].x)


class HeterogeneousSAGEClassifier(nn.Module):
    """Relation-specific GraphSAGE with fixed mean/sum aggregation."""

    def __init__(
        self,
        input_dims: dict[str, int | None],
        num_nodes: dict[str, int],
        metadata: Metadata,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
    ):
        super().__init__()
        self.input_encoder = HeterogeneousInputEncoder(
            input_dims, num_nodes, hidden_channels
        )
        self.layers = nn.ModuleList([
            HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                for edge_type in metadata[1]
            }, aggr="sum")
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_channels)
                for node_type in metadata[0]
            })
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, graph: HeteroData) -> torch.Tensor:
        representations = self.input_encoder(graph)
        for layer, norms in zip(self.layers, self.norms):
            updated = layer(representations, graph.edge_index_dict)
            representations = {
                node_type: norms[node_type](
                    representations[node_type] + torch.relu(updated[node_type])
                ) if node_type in updated else representations[node_type]
                for node_type in representations
            }
        return self.classifier(representations["author"])


class HGTClassifier(nn.Module):
    """Heterogeneous Graph Transformer with type- and relation-aware attention."""

    def __init__(
        self,
        input_dims: dict[str, int | None],
        num_nodes: dict[str, int],
        metadata: Metadata,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        heads: int = 4,
    ):
        super().__init__()
        if hidden_channels % heads != 0:
            raise ValueError("hidden_channels must be divisible by heads")
        self.input_encoder = HeterogeneousInputEncoder(
            input_dims, num_nodes, hidden_channels
        )
        self.layers = nn.ModuleList([
            HGTConv(hidden_channels, hidden_channels, metadata, heads=heads)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_channels)
                for node_type in metadata[0]
            })
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, graph: HeteroData) -> torch.Tensor:
        representations = self.input_encoder(graph)
        for layer, norms in zip(self.layers, self.norms):
            updated = layer(representations, graph.edge_index_dict)
            representations = {
                node_type: norms[node_type](representations[node_type] + updated[node_type])
                if node_type in updated else representations[node_type]
                for node_type in representations
            }
        return self.classifier(representations["author"])
