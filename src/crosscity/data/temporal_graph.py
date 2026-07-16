from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch_geometric.data import TemporalData
from torch_geometric.datasets import JODIEDataset


@dataclass(frozen=True)
class TemporalGraphSplits:
    full: TemporalData
    train: TemporalData
    validation: TemporalData
    test: TemporalData


def load_jodie_wikipedia(
    root: str | Path,
    *,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> TemporalGraphSplits:
    """Load Wikipedia interactions and split the ordered event stream by time."""
    data = JODIEDataset(root=str(Path(root)), name="Wikipedia")[0]
    train, validation, test = data.train_val_test_split(
        val_ratio=validation_ratio, test_ratio=test_ratio
    )
    return TemporalGraphSplits(data, train, validation, test)
