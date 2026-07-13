from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from crosscity.config import DatasetConfig


@dataclass(frozen=True)
class StandardScaler:
    """Scalar statistics fitted on observed values from the training period only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray) -> "StandardScaler":
        observed = values[mask]
        if observed.size == 0:
            raise ValueError("cannot fit a scaler without observed values")
        std = float(observed.std())
        return cls(float(observed.mean()), std if std > 1e-8 else 1.0)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        return values * self.std + self.mean


class TrafficDataset(Dataset):
    """Windowed samples with x [T_in,N,1], y/mask [T_out,N]."""

    def __init__(
        self,
        values: np.ndarray,
        mask: np.ndarray,
        input_steps: int,
        output_steps: int,
        start: int,
        end: int,
    ) -> None:
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.mask = torch.as_tensor(mask, dtype=torch.bool)
        self.input_steps = input_steps
        self.output_steps = output_steps
        # A sample is assigned by its prediction start. This prevents target overlap
        # across splits while still allowing validation/test inputs to use past context.
        self.indices = np.arange(max(start, input_steps), end - output_steps + 1)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        t = int(self.indices[index])
        x = self.values[t - self.input_steps : t].unsqueeze(-1)
        y = self.values[t : t + self.output_steps]
        mask = self.mask[t : t + self.output_steps]
        return x, y, mask


@dataclass(frozen=True)
class DataBundle:
    train: TrafficDataset
    val: TrafficDataset
    test: TrafficDataset
    scaler: StandardScaler
    raw_values: np.ndarray
    raw_mask: np.ndarray
    split_points: tuple[int, int]


def load_speed_matrix(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix in {".h5", ".hdf", ".hdf5"}:
        values = pd.read_hdf(path).to_numpy(dtype=np.float32)
    elif path.suffix == ".npz":
        archive = np.load(path)
        key = "data" if "data" in archive else archive.files[0]
        values = np.asarray(archive[key], dtype=np.float32)
        if values.ndim == 3:
            values = values[..., 0]
    elif path.suffix == ".npy":
        values = np.load(path).astype(np.float32)
    else:
        raise ValueError(f"unsupported data format: {path.suffix}")
    if values.ndim != 2:
        raise ValueError(f"expected [time,nodes], got {values.shape}")
    return values


def build_data_bundle(config: DatasetConfig, few_shot_days: int | None = None) -> DataBundle:
    raw = load_speed_matrix(config.data_path)
    mask = np.isfinite(raw) & (raw > 0)
    clean = np.where(mask, raw, 0.0).astype(np.float32)
    train_end = int(len(clean) * config.train_ratio)
    val_end = int(len(clean) * (config.train_ratio + config.val_ratio))
    scaler = StandardScaler.fit(clean[:train_end], mask[:train_end])
    scaled = scaler.transform(clean).astype(np.float32)
    # Missing inputs become zero in normalized space, a neutral explicit fill value.
    scaled[~mask] = 0.0
    effective_train_end = train_end
    if few_shot_days is not None:
        effective_train_end = min(train_end, config.input_steps + few_shot_days * config.steps_per_day)
    return DataBundle(
        train=TrafficDataset(scaled, mask, config.input_steps, config.output_steps, 0, effective_train_end),
        val=TrafficDataset(scaled, mask, config.input_steps, config.output_steps, train_end, val_end),
        test=TrafficDataset(scaled, mask, config.input_steps, config.output_steps, val_end, len(clean)),
        scaler=scaler,
        raw_values=clean,
        raw_mask=mask,
        split_points=(train_end, val_end),
    )

