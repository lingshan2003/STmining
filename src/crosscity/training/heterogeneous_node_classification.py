from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class HeterogeneousNodeResult:
    best_epoch: int
    validation_accuracy: float
    validation_macro_f1: float
    test_accuracy: float
    test_macro_f1: float
    history: list[dict[str, float]]


@torch.no_grad()
def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    num_classes: int,
) -> tuple[float, float]:
    prediction = logits[mask].argmax(dim=-1)
    target = labels[mask]
    accuracy = float((prediction == target).float().mean())
    f1_scores = []
    for class_id in range(num_classes):
        true_positive = ((prediction == class_id) & (target == class_id)).sum()
        false_positive = ((prediction == class_id) & (target != class_id)).sum()
        false_negative = ((prediction != class_id) & (target == class_id)).sum()
        denominator = 2 * true_positive + false_positive + false_negative
        f1_scores.append(
            float(2 * true_positive / denominator) if denominator else 0.0
        )
    return accuracy, sum(f1_scores) / num_classes


def train_heterogeneous_node_classifier(
    model: nn.Module,
    graph: HeteroData,
    *,
    device: torch.device | str = "cpu",
    learning_rate: float = 0.005,
    weight_decay: float = 1e-4,
    max_epochs: int = 150,
    patience: int = 30,
) -> HeterogeneousNodeResult:
    model = model.to(device)
    graph = graph.to(device)
    labels = graph["author"].y
    num_classes = int(labels.max()) + 1
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    best_f1, best_epoch, best_state, stale = -1.0, 0, None, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(graph)
        loss = F.cross_entropy(
            logits[graph["author"].train_mask], labels[graph["author"].train_mask]
        )
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = model(graph)
        validation_accuracy, validation_f1 = classification_metrics(
            logits, labels, graph["author"].val_mask, num_classes
        )
        history.append({
            "epoch": float(epoch), "loss": float(loss.detach()),
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_f1,
        })
        if validation_f1 > best_f1:
            best_f1, best_epoch = validation_f1, epoch
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
        logits = model(graph)
    validation_accuracy, validation_f1 = classification_metrics(
        logits, labels, graph["author"].val_mask, num_classes
    )
    test_accuracy, test_f1 = classification_metrics(
        logits, labels, graph["author"].test_mask, num_classes
    )
    return HeterogeneousNodeResult(
        best_epoch, validation_accuracy, validation_f1,
        test_accuracy, test_f1, history,
    )
