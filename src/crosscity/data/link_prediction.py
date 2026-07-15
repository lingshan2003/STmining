from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from .citation import NodeClassificationData


@dataclass(frozen=True)
class LinkPredictionSplits:
    train: Data
    validation: Data
    test: Data


def make_link_prediction_splits(
    data: NodeClassificationData,
    *,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.10,
    negative_ratio: float = 1.0,
    seed: int = 42,
) -> LinkPredictionSplits:
    """Hide undirected edges and create equally protected positive/negative examples."""
    graph = Data(x=data.x, edge_index=data.edge_index, num_nodes=data.num_nodes)
    state = torch.random.get_rng_state()
    torch.manual_seed(seed)
    try:
        train, validation, test = RandomLinkSplit(
            num_val=validation_fraction,
            num_test=test_fraction,
            is_undirected=True,
            split_labels=True,
            add_negative_train_samples=True,
            neg_sampling_ratio=negative_ratio,
        )(graph)
    finally:
        torch.random.set_rng_state(state)
    return LinkPredictionSplits(train=train, validation=validation, test=test)
