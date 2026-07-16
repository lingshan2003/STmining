from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn.models.tgn import LastNeighborLoader

from crosscity.data.temporal_graph import TemporalGraphSplits
from crosscity.models.temporal_graph import StaticTemporalLinkPredictor, TGNLinkPredictor


@dataclass(frozen=True)
class TemporalLinkMetrics:
    average_precision: float
    auc: float


@dataclass(frozen=True)
class TemporalTrainingResult:
    best_epoch: int
    validation: TemporalLinkMetrics
    test: TemporalLinkMetrics
    history: list[dict[str, float]]


def _metrics(labels: list[torch.Tensor], scores: list[torch.Tensor]) -> TemporalLinkMetrics:
    label = torch.cat(labels).numpy()
    score = torch.cat(scores).numpy()
    return TemporalLinkMetrics(
        float(average_precision_score(label, score)),
        float(roc_auc_score(label, score)),
    )


def temporal_loaders(
    splits: TemporalGraphSplits,
    *,
    batch_size: int = 200,
) -> tuple[TemporalDataLoader, TemporalDataLoader, TemporalDataLoader]:
    options = {"batch_size": batch_size, "neg_sampling_ratio": 1.0}
    return (
        TemporalDataLoader(splits.train, **options),
        TemporalDataLoader(splits.validation, **options),
        TemporalDataLoader(splits.test, **options),
    )


def _static_epoch(
    model: StaticTemporalLinkPredictor,
    loader: TemporalDataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device | str,
) -> tuple[float, TemporalLinkMetrics]:
    labels, scores, total_loss, total_events = [], [], 0.0, 0
    model.train(optimizer is not None)
    for batch in loader:
        batch = batch.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        positive = model.score(batch.src, batch.dst)
        negative = model.score(batch.src, batch.neg_dst)
        loss = criterion(positive, torch.ones_like(positive))
        loss = loss + criterion(negative, torch.zeros_like(negative))
        if optimizer is not None:
            loss.backward()
            optimizer.step()
        labels.append(torch.cat((torch.ones_like(positive), torch.zeros_like(negative))).cpu())
        scores.append(torch.cat((positive, negative)).sigmoid().detach().cpu())
        total_loss += float(loss.detach()) * batch.num_events
        total_events += batch.num_events
    return total_loss / total_events, _metrics(labels, scores)


def train_static_temporal_baseline(
    model: StaticTemporalLinkPredictor,
    splits: TemporalGraphSplits,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 200,
    learning_rate: float = 1e-3,
    max_epochs: int = 20,
) -> TemporalTrainingResult:
    model = model.to(device)
    train_loader, validation_loader, test_loader = temporal_loaders(
        splits, batch_size=batch_size
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_epoch, best_validation, best_state = 0, None, None
    history = []
    for epoch in range(1, max_epochs + 1):
        loss, _ = _static_epoch(
            model, train_loader, criterion, optimizer, device
        )
        with torch.no_grad():
            _, validation = _static_epoch(
                model, validation_loader, criterion, None, device
            )
        history.append({
            "epoch": float(epoch), "loss": loss,
            "validation_ap": validation.average_precision,
        })
        if best_validation is None or validation.average_precision > best_validation.average_precision:
            best_epoch, best_validation = epoch, validation
            best_state = deepcopy(model.state_dict())
    if best_validation is None or best_state is None:
        raise RuntimeError("training did not produce metrics")
    model.load_state_dict(best_state)
    torch.manual_seed(12345)
    with torch.no_grad():
        _, best_test = _static_epoch(model, test_loader, criterion, None, device)
    return TemporalTrainingResult(
        best_epoch, best_validation, best_test, history
    )


def _tgn_train_epoch(
    model: TGNLinkPredictor,
    full_data: TemporalData,
    loader: TemporalDataLoader,
    neighbor_loader: LastNeighborLoader,
    association: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str,
) -> float:
    model.train()
    model.memory.reset_state()
    neighbor_loader.reset_state()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        node_ids, edge_index, event_ids = neighbor_loader(batch.n_id)
        association[node_ids] = torch.arange(node_ids.size(0), device=device)
        memory, last_update = model.memory(node_ids)
        embedding = model.graph_embedding(
            memory,
            last_update,
            edge_index,
            full_data.t[event_ids],
            full_data.msg[event_ids],
        )
        positive = model.score(
            embedding[association[batch.src]], embedding[association[batch.dst]]
        )
        negative = model.score(
            embedding[association[batch.src]], embedding[association[batch.neg_dst]]
        )
        loss = criterion(positive, torch.ones_like(positive))
        loss = loss + criterion(negative, torch.zeros_like(negative))
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
        loss.backward()
        optimizer.step()
        model.memory.detach()
        total_loss += float(loss.detach()) * batch.num_events
    return total_loss / loader.data.num_events


@torch.no_grad()
def evaluate_tgn(
    model: TGNLinkPredictor,
    full_data: TemporalData,
    loader: TemporalDataLoader,
    neighbor_loader: LastNeighborLoader,
    association: torch.Tensor,
    device: torch.device | str,
) -> TemporalLinkMetrics:
    model.eval()
    labels, scores = [], []
    for batch in loader:
        batch = batch.to(device)
        node_ids, edge_index, event_ids = neighbor_loader(batch.n_id)
        association[node_ids] = torch.arange(node_ids.size(0), device=device)
        memory, last_update = model.memory(node_ids)
        embedding = model.graph_embedding(
            memory,
            last_update,
            edge_index,
            full_data.t[event_ids],
            full_data.msg[event_ids],
        )
        positive = model.score(
            embedding[association[batch.src]], embedding[association[batch.dst]]
        )
        negative = model.score(
            embedding[association[batch.src]], embedding[association[batch.neg_dst]]
        )
        labels.append(torch.cat((torch.ones_like(positive), torch.zeros_like(negative))).cpu())
        scores.append(torch.cat((positive, negative)).sigmoid().cpu())
        model.memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    return _metrics(labels, scores)


def train_tgn(
    model: TGNLinkPredictor,
    splits: TemporalGraphSplits,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 200,
    neighbor_size: int = 10,
    learning_rate: float = 1e-4,
    max_epochs: int = 20,
) -> TemporalTrainingResult:
    model = model.to(device)
    full_data = splits.full.to(device)
    train_loader, validation_loader, test_loader = temporal_loaders(
        splits, batch_size=batch_size
    )
    neighbor_loader = LastNeighborLoader(
        full_data.num_nodes, size=neighbor_size, device=device
    )
    association = torch.empty(full_data.num_nodes, dtype=torch.long, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    best_epoch, best_validation, best_state = 0, None, None
    history = []
    for epoch in range(1, max_epochs + 1):
        loss = _tgn_train_epoch(
            model, full_data, train_loader, neighbor_loader,
            association, optimizer, criterion, device,
        )
        torch.manual_seed(12345)
        validation = evaluate_tgn(
            model, full_data, validation_loader, neighbor_loader, association, device
        )
        history.append({
            "epoch": float(epoch), "loss": loss,
            "validation_ap": validation.average_precision,
        })
        if best_validation is None or validation.average_precision > best_validation.average_precision:
            best_epoch, best_validation = epoch, validation
            best_state = deepcopy(model.state_dict())
    if best_validation is None or best_state is None:
        raise RuntimeError("training did not produce metrics")
    model.load_state_dict(best_state)
    model.memory.reset_state()
    neighbor_loader.reset_state()
    torch.manual_seed(12345)
    # Replay past events without optimization to reconstruct memory exactly at
    # the validation/test boundaries under the selected checkpoint.
    evaluate_tgn(
        model, full_data, train_loader, neighbor_loader, association, device
    )
    best_validation = evaluate_tgn(
        model, full_data, validation_loader, neighbor_loader, association, device
    )
    best_test = evaluate_tgn(
        model, full_data, test_loader, neighbor_loader, association, device
    )
    return TemporalTrainingResult(best_epoch, best_validation, best_test, history)
