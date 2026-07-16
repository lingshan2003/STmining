from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from crosscity.data.heterogeneous_recommendation import (
    HeterogeneousRecommendationData,
    sample_product_negatives,
)


@dataclass(frozen=True)
class HeterogeneousRankingMetrics:
    recall: float
    ndcg: float


@dataclass(frozen=True)
class HeterogeneousRecommendationResult:
    best_epoch: int
    validation: HeterogeneousRankingMetrics
    test: HeterogeneousRankingMetrics
    history: list[dict[str, float]]


def pairwise_ranking_loss(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    return -F.logsigmoid(positive - negative).mean()


@torch.no_grad()
def heterogeneous_ranking_metrics(
    model: nn.Module,
    data: HeterogeneousRecommendationData,
    *,
    split: str,
    k: int = 3,
) -> HeterogeneousRankingMetrics:
    model.eval()
    representations = model.encode(data.graph)
    if split == "validation":
        users, targets = data.validation_users, data.validation_products
        seen_users, seen_products = data.train_users, data.train_products
    elif split == "test":
        users, targets = data.test_users, data.test_products
        seen_users = torch.cat((data.train_users, data.validation_users))
        seen_products = torch.cat((data.train_products, data.validation_products))
    else:
        raise ValueError("split must be 'validation' or 'test'")

    scores = representations["user"][users] @ representations["product"].t()
    row_by_user = {int(user): row for row, user in enumerate(users.tolist())}
    for user, product in zip(seen_users.tolist(), seen_products.tolist()):
        if user in row_by_user:
            scores[row_by_user[user], product] = -torch.inf
    ranked = scores.topk(min(k, data.num_products), dim=1).indices
    hits = ranked == targets[:, None]
    recall = float(hits.any(dim=1).float().mean())
    discounts = 1 / torch.log2(
        torch.arange(ranked.size(1), device=ranked.device, dtype=torch.float) + 2
    )
    ndcg = float((hits.float() * discounts).sum(dim=1).mean())
    return HeterogeneousRankingMetrics(recall, ndcg)


def train_heterogeneous_recommender(
    model: nn.Module,
    data: HeterogeneousRecommendationData,
    *,
    device: torch.device | str = "cpu",
    learning_rate: float = 0.01,
    max_epochs: int = 300,
    patience: int = 60,
    k: int = 3,
    seed: int = 42,
) -> HeterogeneousRecommendationResult:
    model = model.to(device)
    device_data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    best_ndcg, best_epoch, best_state, stale = -1.0, 0, None, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        negative = sample_product_negatives(
            data.train_users,
            data.num_products,
            data.known_purchases,
            generator=generator,
        ).to(device)
        optimizer.zero_grad()
        representations = model.encode(device_data.graph)
        positive_score = model.score(
            representations, device_data.train_users, device_data.train_products
        )
        negative_score = model.score(
            representations, device_data.train_users, negative
        )
        loss = pairwise_ranking_loss(positive_score, negative_score)
        loss.backward()
        optimizer.step()
        validation = heterogeneous_ranking_metrics(
            model, device_data, split="validation", k=k
        )
        history.append({
            "epoch": float(epoch), "loss": float(loss.detach()),
            "validation_ndcg": validation.ndcg,
        })
        if validation.ndcg > best_ndcg:
            best_ndcg, best_epoch = validation.ndcg, epoch
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return HeterogeneousRecommendationResult(
        best_epoch,
        heterogeneous_ranking_metrics(model, device_data, split="validation", k=k),
        heterogeneous_ranking_metrics(model, device_data, split="test", k=k),
        history,
    )
