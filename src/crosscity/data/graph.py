from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch


def load_adjacency(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        adjacency = np.load(path)
    elif path.suffix == ".npz":
        archive = np.load(path)
        key = "adjacency" if "adjacency" in archive else archive.files[0]
        adjacency = archive[key]
    elif path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle, encoding="latin1")
        # DCRNN files contain (sensor_ids, id_to_index, adjacency).
        adjacency = payload[-1] if isinstance(payload, (tuple, list)) else payload
    else:
        raise ValueError(f"unsupported adjacency format: {path.suffix}")
    adjacency = np.asarray(adjacency, dtype=np.float32)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"adjacency must be square, got {adjacency.shape}")
    return adjacency


def gaussian_adjacency(distances: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    finite = distances[np.isfinite(distances)]
    sigma = float(finite.std())
    weights = np.exp(-np.square(distances / max(sigma, 1e-8)))
    weights[~np.isfinite(weights)] = 0
    weights[weights < threshold] = 0
    np.fill_diagonal(weights, 0)
    return weights.astype(np.float32)


def normalize_adjacency(adjacency: np.ndarray | torch.Tensor) -> torch.Tensor:
    a = torch.as_tensor(adjacency, dtype=torch.float32)
    a = a + torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
    degree = a.sum(dim=1).clamp_min(1e-8)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt[:, None] * a * inv_sqrt[None, :]

