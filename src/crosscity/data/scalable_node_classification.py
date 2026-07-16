from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.transforms import ToUndirected


def load_ogbn_arxiv(root: str | Path) -> Data:
    """Load OGBN-Arxiv and expose its official chronological split as masks.

    OGB is imported lazily so the rest of the package remains importable before
    the optional public-data dependency has been installed.
    """
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise ImportError(
            "ogb is required for OGBN-Arxiv; install the project again with "
            "`python -m pip install -e '.[dev]'`"
        ) from exc

    # OGB stores a full PyG ``Data`` object rather than a tensor-only state dict.
    # Since PyTorch 2.6, torch.load defaults to weights_only=True, so explicitly
    # allow the PyG container classes required by this trusted public dataset.
    with torch.serialization.safe_globals([DataEdgeAttr, DataTensorAttr, GlobalStorage]):
        dataset = PygNodePropPredDataset(name="ogbn-arxiv", root=str(Path(root)))
    graph = ToUndirected()(dataset[0])
    graph.y = graph.y.view(-1).long()
    split = dataset.get_idx_split()
    for split_name, indices in split.items():
        mask = torch.zeros(graph.num_nodes, dtype=torch.bool)
        mask[indices.view(-1)] = True
        setattr(graph, f"{split_name}_mask", mask)
    return graph


def make_node_masks(
    num_nodes: int,
    train_indices: torch.Tensor,
    validation_indices: torch.Tensor,
    test_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build boolean masks; kept separate so split invariants are easy to test."""
    masks = []
    for indices in (train_indices, validation_indices, test_indices):
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[indices] = True
        masks.append(mask)
    return masks[0], masks[1], masks[2]
