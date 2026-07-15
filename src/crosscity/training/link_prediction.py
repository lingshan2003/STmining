from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import VGAE


@dataclass(frozen=True)
class LinkMetrics:
    auc: float
    average_precision: float


@dataclass(frozen=True)
class LinkPredictionResult:
    best_epoch: int
    validation: LinkMetrics
    test: LinkMetrics
    history: list[dict[str, float]]


@torch.no_grad()
def evaluate_link_model(model: nn.Module, data: Data) -> LinkMetrics:
    model.eval()
    z = model.encode(data.x, data.edge_index)
    positive = model.decoder(z, data.pos_edge_label_index, sigmoid=True)
    negative = model.decoder(z, data.neg_edge_label_index, sigmoid=True)
    labels = torch.cat((torch.ones_like(positive), torch.zeros_like(negative))).cpu().numpy()
    scores = torch.cat((positive, negative)).cpu().numpy()
    return LinkMetrics(
        auc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
    )


@torch.no_grad()
def feature_similarity_baseline(data: Data) -> LinkMetrics:
    def score(edge_index: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        return torch.nn.functional.cosine_similarity(data.x[source], data.x[target])

    positive = score(data.pos_edge_label_index)
    negative = score(data.neg_edge_label_index)
    labels = torch.cat((torch.ones_like(positive), torch.zeros_like(negative))).cpu().numpy()
    scores = torch.cat((positive, negative)).cpu().numpy()
    return LinkMetrics(
        auc=float(roc_auc_score(labels, scores)),
        average_precision=float(average_precision_score(labels, scores)),
    )


def train_link_predictor(
    model: nn.Module,
    train_data: Data,
    validation_data: Data,
    test_data: Data,
    *,
    device: torch.device | str = "cpu",
    learning_rate: float = 0.01,
    max_epochs: int = 300,
    patience: int = 50,
) -> LinkPredictionResult:
    model = model.to(device)
    train_data = train_data.to(device)
    validation_data = validation_data.to(device)
    test_data = test_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_ap, best_epoch, best_state, stale = -1.0, 0, None, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z = model.encode(train_data.x, train_data.edge_index)
        loss = model.recon_loss(
            z, train_data.pos_edge_label_index, train_data.neg_edge_label_index
        )
        if isinstance(model, VGAE):
            loss = loss + model.kl_loss() / train_data.num_nodes
        loss.backward()
        optimizer.step()

        validation = evaluate_link_model(model, validation_data)
        history.append({
            "epoch": float(epoch),
            "loss": float(loss.detach()),
            "val_auc": validation.auc,
            "val_ap": validation.average_precision,
        })
        if validation.average_precision > best_ap:
            best_ap, best_epoch = validation.average_precision, epoch
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return LinkPredictionResult(
        best_epoch=best_epoch,
        validation=evaluate_link_model(model, validation_data),
        test=evaluate_link_model(model, test_data),
        history=history,
    )
