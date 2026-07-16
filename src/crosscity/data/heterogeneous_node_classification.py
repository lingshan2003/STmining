from __future__ import annotations

from pathlib import Path

from torch_geometric.data import HeteroData
from torch_geometric.datasets import DBLP


def load_dblp(root: str | Path) -> HeteroData:
    """Load the public DBLP heterogeneous author-classification graph."""
    return DBLP(root=str(Path(root)))[0]
