from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Subset
from torch_geometric.datasets import TUDataset


@dataclass(frozen=True)
class GraphDatasetSplits:
    train: Subset
    validation: Subset
    test: Subset
    num_features: int
    num_classes: int


def load_tu_dataset(root: str | Path, name: str = "MUTAG") -> TUDataset:
    """Load a TU graph-classification dataset through PyTorch Geometric."""
    return TUDataset(root=str(Path(root)), name=name.upper(), use_node_attr=True)


def stratified_graph_split(
    dataset: TUDataset,
    *,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> GraphDatasetSplits:
    """Create deterministic, disjoint splits while preserving every class."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must lie strictly between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than one")

    labels = torch.tensor([int(graph.y.item()) for graph in dataset])
    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for label in labels.unique(sorted=True):
        class_indices = torch.where(labels == label)[0]
        class_indices = class_indices[torch.randperm(len(class_indices), generator=generator)]
        train_end = max(1, int(len(class_indices) * train_fraction))
        validation_end = train_end + max(1, int(len(class_indices) * validation_fraction))
        validation_end = min(validation_end, len(class_indices) - 1)
        train_indices.extend(class_indices[:train_end].tolist())
        validation_indices.extend(class_indices[train_end:validation_end].tolist())
        test_indices.extend(class_indices[validation_end:].tolist())

    return GraphDatasetSplits(
        train=Subset(dataset, train_indices),
        validation=Subset(dataset, validation_indices),
        test=Subset(dataset, test_indices),
        num_features=dataset.num_features,
        num_classes=dataset.num_classes,
    )
