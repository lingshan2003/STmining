from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import RGCNConv


class KnowledgeGraphEmbedding(nn.Module):
    """Entity/relation embeddings without message passing."""

    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.entity = nn.Embedding(num_entities, embedding_dim)
        self.relation = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        del edge_index, edge_type
        return self.entity.weight

    def score(
        self, entity: torch.Tensor, triples: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class TransE(KnowledgeGraphEmbedding):
    """Score a triple by how closely head + relation reaches tail."""

    def score(self, entity: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        head, relation, tail = triples
        distance = entity[head] + self.relation(relation) - entity[tail]
        return -distance.norm(p=1, dim=-1)


class DistMult(KnowledgeGraphEmbedding):
    """Score a triple with a relation-conditioned tri-linear product."""

    def score(self, entity: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        head, relation, tail = triples
        return (entity[head] * self.relation(relation) * entity[tail]).sum(dim=-1)


class RGCNDistMult(nn.Module):
    """Encode entities with relation-aware messages and decode with DistMult."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int,
        num_bases: int | None = None,
    ):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.conv = RGCNConv(
            num_entities,
            embedding_dim,
            2 * num_relations,
            num_bases=num_bases,
        )
        self.relation = nn.Embedding(num_relations, embedding_dim)
        nn.init.xavier_uniform_(self.relation.weight)

    def encode(self, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.conv(None, edge_index, edge_type))

    def score(self, entity: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        head, relation, tail = triples
        return (entity[head] * self.relation(relation) * entity[tail]).sum(dim=-1)
