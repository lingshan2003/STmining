from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import add_self_loops


@dataclass(frozen=True)
class NodeClassificationData:
    x: torch.Tensor
    y: torch.Tensor
    edge_index: torch.Tensor
    train_mask: torch.Tensor
    val_mask: torch.Tensor
    test_mask: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_features(self) -> int:
        return self.x.shape[1]

    @property
    def num_classes(self) -> int:
        return int(self.y.max().item()) + 1

    def to(self, device: torch.device | str) -> NodeClassificationData:
        return NodeClassificationData(**{
            name: getattr(self, name).to(device)
            for name in self.__dataclass_fields__
        })


def load_planetoid(root: str | Path, name: str = "cora") -> NodeClassificationData:
    """Download (when needed) and load a public Planetoid split through PyG."""
    canonical_names = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
    key = name.lower()
    if key not in canonical_names:
        raise ValueError(f"unsupported Planetoid dataset: {name}")
    dataset = Planetoid(
        root=str(Path(root)), name=canonical_names[key], transform=NormalizeFeatures()
    )
    graph = dataset[0]
    return NodeClassificationData(
        x=graph.x,
        y=graph.y,
        edge_index=graph.edge_index,
        train_mask=graph.train_mask,
        val_mask=graph.val_mask,
        test_mask=graph.test_mask,
    )


def identity_edges(num_nodes: int) -> torch.Tensor:
    empty = torch.empty((2, 0), dtype=torch.long)
    return add_self_loops(empty, num_nodes=num_nodes)[0]


def shuffled_edges(edge_index: torch.Tensor, num_nodes: int, seed: int = 42) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    permutation = torch.randperm(num_nodes, generator=generator).to(edge_index.device)
    return permutation[edge_index]
