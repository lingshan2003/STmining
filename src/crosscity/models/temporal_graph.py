from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator


class StaticTemporalLinkPredictor(nn.Module):
    """A time-agnostic node embedding baseline for temporal link prediction."""

    def __init__(self, num_nodes: int, embedding_dim: int = 100):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embedding_dim)
        self.predictor = LinkPredictor(embedding_dim)
        nn.init.xavier_uniform_(self.embedding.weight)

    def score(self, source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.embedding(source), self.embedding(destination))


class TemporalGraphEmbedding(nn.Module):
    """Attend over recent neighbors using edge messages and elapsed time."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        message_dim: int,
        time_encoder: nn.Module,
    ):
        super().__init__()
        self.time_encoder = time_encoder
        edge_dim = message_dim + time_encoder.out_channels
        self.convolution = TransformerConv(
            in_channels,
            out_channels // 2,
            heads=2,
            dropout=0.1,
            edge_dim=edge_dim,
        )

    def forward(
        self,
        memory: torch.Tensor,
        last_update: torch.Tensor,
        edge_index: torch.Tensor,
        event_time: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        elapsed = last_update[edge_index[0]] - event_time
        encoded_time = self.time_encoder(elapsed.to(memory.dtype))
        edge_attribute = torch.cat((encoded_time, message), dim=-1)
        return self.convolution(memory, edge_index, edge_attribute)


class LinkPredictor(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.source = nn.Linear(in_channels, in_channels)
        self.destination = nn.Linear(in_channels, in_channels)
        self.output = nn.Linear(in_channels, 1)

    def forward(self, source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.source(source) + self.destination(destination))
        return self.output(hidden).squeeze(-1)


class TGNLinkPredictor(nn.Module):
    """TGN memory, temporal neighborhood attention, and a link decoder."""

    def __init__(
        self,
        num_nodes: int,
        message_dim: int,
        memory_dim: int = 100,
        time_dim: int = 100,
        embedding_dim: int = 100,
    ):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes,
            message_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(message_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.graph_embedding = TemporalGraphEmbedding(
            memory_dim, embedding_dim, message_dim, self.memory.time_enc
        )
        self.predictor = LinkPredictor(embedding_dim)

    def score(self, source: torch.Tensor, destination: torch.Tensor) -> torch.Tensor:
        return self.predictor(source, destination)
