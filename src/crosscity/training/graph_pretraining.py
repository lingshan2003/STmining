from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import torch
from torch.nn import functional as F
from torch_geometric.utils import dropout_edge

from crosscity.data.citation import NodeClassificationData
from crosscity.models.graph_pretraining import GraphEncoder, GraphMAE, symmetric_contrastive_loss


def few_label_training_data(
    data: NodeClassificationData, labels_per_class: int, seed: int = 42
) -> NodeClassificationData:
    """Restrict only the training mask; validation and test remain untouched."""
    if labels_per_class < 1:
        raise ValueError("labels_per_class must be positive")
    generator = torch.Generator().manual_seed(seed)
    mask = torch.zeros_like(data.train_mask)
    for label in range(data.num_classes):
        candidates = torch.where(data.train_mask & (data.y == label))[0]
        if len(candidates) < labels_per_class:
            raise ValueError(f"class {label} has only {len(candidates)} training labels")
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:labels_per_class]]
        mask[chosen] = True
    return replace(data, train_mask=mask)


def random_node_mask(num_nodes: int, ratio: float, device: torch.device | str) -> torch.Tensor:
    if not 0 < ratio < 1:
        raise ValueError("mask ratio must lie strictly between zero and one")
    count = max(1, int(num_nodes * ratio))
    indices = torch.randperm(num_nodes, device=device)[:count]
    mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    mask[indices] = True
    return mask


def pretrain_graphmae(
    model: GraphMAE,
    data: NodeClassificationData,
    *,
    mask_ratio: float = 0.4,
    learning_rate: float = 0.001,
    epochs: int = 200,
) -> list[dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        mask = random_node_mask(data.num_nodes, mask_ratio, data.x.device)
        prediction, _ = model(data.x, data.edge_index, mask)
        loss = F.mse_loss(prediction[mask], data.x[mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"epoch": float(epoch), "loss": float(loss.detach())})
    return history


def _feature_dropout(x: torch.Tensor, probability: float) -> torch.Tensor:
    keep = torch.rand(x.shape[1], device=x.device) >= probability
    return x * keep.to(x.dtype)


def pretrain_contrastive(
    encoder: GraphEncoder,
    data: NodeClassificationData,
    *,
    feature_drop: float = 0.2,
    edge_drop: float = 0.2,
    sample_size: int = 512,
    temperature: float = 0.2,
    learning_rate: float = 0.001,
    epochs: int = 200,
) -> list[dict[str, float]]:
    """GRACE-style full-graph views with sampled anchors for bounded InfoNCE memory."""
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    history = []
    for epoch in range(1, epochs + 1):
        encoder.train()
        edge_first, _ = dropout_edge(data.edge_index, p=edge_drop, training=True)
        edge_second, _ = dropout_edge(data.edge_index, p=edge_drop, training=True)
        first = encoder(_feature_dropout(data.x, feature_drop), edge_first)
        second = encoder(_feature_dropout(data.x, feature_drop), edge_second)
        anchors = torch.randperm(data.num_nodes, device=data.x.device)[:sample_size]
        loss = symmetric_contrastive_loss(first[anchors], second[anchors], temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append({"epoch": float(epoch), "loss": float(loss.detach())})
    return history


def clone_encoder(encoder: GraphEncoder) -> GraphEncoder:
    """Deep-copy a pretrained encoder so downstream runs cannot contaminate each other."""
    return deepcopy(encoder)
