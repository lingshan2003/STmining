from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.loader import DataLoader


@dataclass(frozen=True)
class GraphClassificationResult:
    best_epoch: int
    validation_accuracy: float
    test_accuracy: float
    history: list[dict[str, float]]


@torch.no_grad()
def graph_accuracy(model: nn.Module, loader: DataLoader, device: torch.device | str) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        prediction = model(batch.x, batch.edge_index, batch.batch).argmax(-1)
        correct += int((prediction == batch.y.view(-1)).sum())
        total += batch.num_graphs
    return correct / max(total, 1)


def train_graph_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    *,
    device: torch.device | str = "cpu",
    learning_rate: float = 0.001,
    weight_decay: float = 5e-4,
    max_epochs: int = 300,
    patience: int = 50,
) -> GraphClassificationResult:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_accuracy, best_epoch, best_state, stale = -1.0, 0, None, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        graph_count = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, batch.batch)
            loss = F.cross_entropy(logits, batch.y.view(-1))
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * batch.num_graphs
            graph_count += batch.num_graphs
        validation_accuracy = graph_accuracy(model, validation_loader, device)
        history.append({
            "epoch": float(epoch),
            "loss": loss_sum / max(graph_count, 1),
            "val_accuracy": validation_accuracy,
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
    return GraphClassificationResult(
        best_epoch=best_epoch,
        validation_accuracy=best_accuracy,
        test_accuracy=graph_accuracy(model, test_loader, device),
        history=history,
    )
