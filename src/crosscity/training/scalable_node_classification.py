from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import time

import torch
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from crosscity.models.scalable_gnn import SampledGraphSAGE, ScalableMLP


@dataclass(frozen=True)
class ScalableNodeResult:
    best_epoch: int
    validation_accuracy: float
    test_accuracy: float
    history: list[dict[str, float]]
    seconds: float
    peak_cuda_mb: float


def _sampling_backend_available() -> bool:
    try:
        import pyg_lib  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import torch_sparse  # noqa: F401
        return True
    except ImportError:
        return False


def require_sampling_backend() -> None:
    """Fail early with an actionable error instead of inside the first epoch."""
    if not _sampling_backend_available():
        raise RuntimeError(
            "NeighborLoader needs pyg-lib or torch-sparse. Install the wheel matching "
            "the server's PyTorch and CUDA versions from https://data.pyg.org/whl/."
        )


def make_neighbor_loaders(
    graph: Data,
    *,
    num_neighbors: list[int],
    batch_size: int,
    inference_batch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[NeighborLoader, NeighborLoader]:
    """Create a stochastic training loader and exact layer-wise inference loader."""
    require_sampling_backend()
    train_loader = NeighborLoader(
        graph,
        input_nodes=graph.train_mask,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    inference_loader = NeighborLoader(
        graph,
        input_nodes=None,
        num_neighbors=[-1],
        batch_size=inference_batch_size or batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )
    return train_loader, inference_loader


def node_accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return float((logits[mask].argmax(dim=-1) == labels[mask]).float().mean())


def _peak_cuda_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024**2


def train_sampled_graphsage(
    model: SampledGraphSAGE,
    graph: Data,
    *,
    device: torch.device | str = "cpu",
    num_neighbors: list[int] | None = None,
    batch_size: int = 1024,
    inference_batch_size: int = 4096,
    num_workers: int = 0,
    learning_rate: float = 0.003,
    weight_decay: float = 0.0,
    max_epochs: int = 30,
    patience: int = 8,
) -> ScalableNodeResult:
    """Train on sampled neighborhoods and select checkpoints using validation only."""
    device = torch.device(device)
    fanouts = num_neighbors or [15] * model.num_layers
    if len(fanouts) != model.num_layers:
        raise ValueError("num_neighbors must contain one fanout per GNN layer")
    train_loader, inference_loader = make_neighbor_loaders(
        graph,
        num_neighbors=fanouts,
        batch_size=batch_size,
        inference_batch_size=inference_batch_size,
        num_workers=num_workers,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_accuracy, best_epoch, best_state, stale = -1.0, 0, None, 0
    history: list[dict[str, float]] = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        seed_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index)[: batch.batch_size]
            labels = batch.y[: batch.batch_size]
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * batch.batch_size
            seed_count += batch.batch_size

        logits = model.layerwise_inference(graph.x, inference_loader, device)
        validation_accuracy = node_accuracy(logits, graph.y, graph.valid_mask)
        history.append({
            "epoch": float(epoch),
            "loss": loss_sum / seed_count,
            "validation_accuracy": validation_accuracy,
        })
        if validation_accuracy > best_accuracy:
            best_accuracy, best_epoch = validation_accuracy, epoch
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    logits = model.layerwise_inference(graph.x, inference_loader, device)
    return ScalableNodeResult(
        best_epoch=best_epoch,
        validation_accuracy=node_accuracy(logits, graph.y, graph.valid_mask),
        test_accuracy=node_accuracy(logits, graph.y, graph.test_mask),
        history=history,
        seconds=time.perf_counter() - started,
        peak_cuda_mb=_peak_cuda_mb(device),
    )


def train_scalable_mlp(
    model: ScalableMLP,
    graph: Data,
    *,
    device: torch.device | str = "cpu",
    learning_rate: float = 0.003,
    weight_decay: float = 0.0,
    max_epochs: int = 100,
    patience: int = 15,
) -> ScalableNodeResult:
    """Train a feature-only baseline without moving the graph edges to the device."""
    device = torch.device(device)
    model = model.to(device)
    x, y = graph.x.to(device), graph.y.to(device)
    masks = {
        name: getattr(graph, f"{name}_mask").to(device)
        for name in ("train", "valid", "test")
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    best_accuracy, best_epoch, best_state, stale = -1.0, 0, None, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits[masks["train"]], y[masks["train"]])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x)
        validation_accuracy = node_accuracy(logits, y, masks["valid"])
        history.append({
            "epoch": float(epoch), "loss": float(loss.detach()),
            "validation_accuracy": validation_accuracy,
        })
        if validation_accuracy > best_accuracy:
            best_accuracy, best_epoch = validation_accuracy, epoch
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(x)
    return ScalableNodeResult(
        best_epoch=best_epoch,
        validation_accuracy=node_accuracy(logits, y, masks["valid"]),
        test_accuracy=node_accuracy(logits, y, masks["test"]),
        history=history,
        seconds=time.perf_counter() - started,
        peak_cuda_mb=_peak_cuda_mb(device),
    )
