from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.datasets import RelLinkPredDataset


@dataclass(frozen=True)
class KnowledgeGraphData:
    num_entities: int
    num_relations: int
    train: torch.Tensor
    validation: torch.Tensor
    test: torch.Tensor
    edge_index: torch.Tensor
    edge_type: torch.Tensor
    entity_names: tuple[str, ...] | None = None
    relation_names: tuple[str, ...] | None = None

    def to(self, device: torch.device | str) -> KnowledgeGraphData:
        values = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return KnowledgeGraphData(**values)

    @property
    def all_true_triples(self) -> set[tuple[int, int, int]]:
        triples = torch.cat((self.train, self.validation, self.test), dim=1).cpu().t()
        return {tuple(map(int, triple)) for triple in triples.tolist()}


def _message_graph(train: torch.Tensor, num_relations: int) -> tuple[torch.Tensor, torch.Tensor]:
    head, relation, tail = train
    edge_index = torch.cat((torch.stack((head, tail)), torch.stack((tail, head))), dim=1)
    edge_type = torch.cat((relation, relation + num_relations))
    return edge_index, edge_type


def make_toy_knowledge_graph() -> KnowledgeGraphData:
    """Create a small geographic KG whose missing facts require relation semantics."""
    entities = ("Alice", "Bob", "Carol", "Paris", "London", "France", "UK")
    relations = ("lives_in", "located_in", "friend_of")
    entity = {name: index for index, name in enumerate(entities)}
    relation = {name: index for index, name in enumerate(relations)}

    def triples(rows: list[tuple[str, str, str]]) -> torch.Tensor:
        return torch.tensor([
            [entity[head] for head, _, _ in rows],
            [relation[edge] for _, edge, _ in rows],
            [entity[tail] for _, _, tail in rows],
        ])

    train = triples([
        ("Alice", "lives_in", "Paris"),
        ("Bob", "lives_in", "London"),
        ("Paris", "located_in", "France"),
        ("London", "located_in", "UK"),
        ("Alice", "friend_of", "Carol"),
        ("Carol", "friend_of", "Alice"),
        ("Bob", "friend_of", "Carol"),
    ])
    validation = triples([("Carol", "lives_in", "Paris")])
    test = triples([
        ("Carol", "located_in", "France"),
        ("Carol", "friend_of", "Bob"),
    ])
    edge_index, edge_type = _message_graph(train, len(relations))
    return KnowledgeGraphData(
        len(entities), len(relations), train, validation, test, edge_index, edge_type,
        entities, relations,
    )


def load_fb15k237(root: str | Path) -> KnowledgeGraphData:
    """Load the public FB15k-237 split provided by PyTorch Geometric."""
    graph = RelLinkPredDataset(root=str(Path(root)), name="FB15k-237")[0]

    def triples(split: str) -> torch.Tensor:
        edge_index = getattr(graph, f"{split}_edge_index")
        edge_type = getattr(graph, f"{split}_edge_type")
        return torch.stack((edge_index[0], edge_type, edge_index[1]))

    train = triples("train")
    validation = triples("valid")
    test = triples("test")
    num_relations = int(graph.train_edge_type.max()) + 1
    return KnowledgeGraphData(
        graph.num_nodes, num_relations, train, validation, test,
        graph.edge_index, graph.edge_type,
    )


def sample_corrupted_triples(
    positive: torch.Tensor,
    num_entities: int,
    known: set[tuple[int, int, int]],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Create negatives by replacing the head or tail with a random entity."""
    negative = positive.clone().cpu()
    corrupt_head = torch.rand(positive.size(1), generator=generator) < 0.5
    for column in range(positive.size(1)):
        while True:
            candidate = int(torch.randint(num_entities, (1,), generator=generator))
            row = 0 if bool(corrupt_head[column]) else 2
            negative[row, column] = candidate
            if tuple(map(int, negative[:, column])) not in known:
                break
    return negative
