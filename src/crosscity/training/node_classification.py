from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from crosscity.data.citation import NodeClassificationData


@dataclass(frozen=True)
class NodeClassificationResult:
    best_epoch: int
    validation_accuracy: float
    test_accuracy: float
    history: list[dict[str, float]]


@torch.no_grad()
def accuracy(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    return float((logits[mask].argmax(-1) == labels[mask]).float().mean())


def train_node_classifier(
    model: nn.Module,
    data: NodeClassificationData,
    *,
    learning_rate: float = 0.01,
    weight_decay: float = 5e-4,
    max_epochs: int = 300,
    patience: int = 50,
) -> NodeClassificationResult:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_accuracy, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
            validation_accuracy = accuracy(logits, data.y, data.val_mask)
        history.append({
            "epoch": float(epoch),
            "loss": float(loss.detach()),
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
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    return NodeClassificationResult(
        best_epoch=best_epoch,
        validation_accuracy=best_accuracy,
        test_accuracy=accuracy(logits, data.y, data.test_mask),
        history=history,
    )
