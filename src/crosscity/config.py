from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    city: str
    data_path: str
    adjacency_path: str
    input_steps: int = 12
    output_steps: int = 12
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    steps_per_day: int = 288

    def __post_init__(self) -> None:
        if self.input_steps < 1 or self.output_steps < 1:
            raise ValueError("input_steps and output_steps must be positive")
        if not 0 < self.train_ratio < 1 or not 0 <= self.val_ratio < 1:
            raise ValueError("invalid train/validation ratios")
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError("train_ratio + val_ratio must be less than one")


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "stgcn"
    mode: str = "target-full"
    seed: int = 42
    batch_size: int = 32
    hidden_dim: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 8
    amp: bool = True
    few_shot_days: int | None = None


@dataclass(frozen=True)
class ProjectConfig:
    dataset: DatasetConfig
    experiment: ExperimentConfig


def _known_values(cls: type, raw: dict[str, Any]) -> dict[str, Any]:
    names = {item.name for item in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return raw


def load_config(path: str | Path) -> ProjectConfig:
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    dataset_values = _known_values(DatasetConfig, raw["dataset"])
    # Repository configs live in <project>/configs. Resolve their data paths once
    # here so library calls behave identically from the CLI, notebooks, and tests.
    project_root = path.parent.parent if path.parent.name == "configs" else path.parent
    for key in ("data_path", "adjacency_path"):
        value = Path(dataset_values[key]).expanduser()
        if not value.is_absolute():
            dataset_values[key] = str((project_root / value).resolve())
    return ProjectConfig(
        dataset=DatasetConfig(**dataset_values),
        experiment=ExperimentConfig(**_known_values(ExperimentConfig, raw["experiment"])),
    )
