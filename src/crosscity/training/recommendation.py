from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn

from crosscity.data.recommendation import (
    RecommendationData,
    interaction_pairs,
    sample_bpr_negatives,
)


@dataclass(frozen=True)
class RankingMetrics:
    recall: float
    ndcg: float


@dataclass(frozen=True)
class RecommendationResult:
    best_epoch: int
    validation: RankingMetrics
    test: RankingMetrics
    history: list[dict[str, float]]


@dataclass(frozen=True)
class RankingDiagnostics:
    """Accuracy and exposure diagnostics for a top-k recommendation list."""

    metrics: RankingMetrics
    coverage: float
    average_popularity: float
    head_recall: float
    tail_recall: float


def bpr_loss(
    embeddings: torch.Tensor,
    users: torch.Tensor,
    positive_items: torch.Tensor,
    negative_items: torch.Tensor,
    num_users: int,
    regularization: float = 1e-4,
) -> torch.Tensor:
    user_embedding = embeddings[users]
    positive_embedding = embeddings[positive_items + num_users]
    negative_embedding = embeddings[negative_items + num_users]
    positive_score = (user_embedding * positive_embedding).sum(-1)
    negative_score = (user_embedding * negative_embedding).sum(-1)
    ranking = -torch.nn.functional.logsigmoid(positive_score - negative_score).mean()
    penalty = (
        user_embedding.square().sum()
        + positive_embedding.square().sum()
        + negative_embedding.square().sum()
    ) / len(users)
    return ranking + regularization * penalty


def _ground_truth(users: torch.Tensor, items: torch.Tensor) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for user, item in zip(users.tolist(), items.tolist()):
        result.setdefault(user, set()).add(item)
    return result


@torch.no_grad()
def ranking_diagnostics(
    embeddings: torch.Tensor,
    data: RecommendationData,
    *,
    split: str,
    k: int = 20,
    head_fraction: float = 0.2,
) -> RankingDiagnostics:
    """Measure accuracy, catalogue coverage, popularity and head/tail recall."""
    if not 0 < head_fraction < 1:
        raise ValueError("head_fraction must be between 0 and 1")
    if split == "validation":
        target_users, target_items = data.validation_users, data.validation_items
        seen_users, seen_items = data.train_users, data.train_items
    elif split == "test":
        target_users, target_items = data.test_users, data.test_items
        seen_users = torch.cat((data.train_users, data.validation_users))
        seen_items = torch.cat((data.train_items, data.validation_items))
    else:
        raise ValueError("split must be 'validation' or 'test'")

    truth = _ground_truth(target_users.cpu(), target_items.cpu())
    users = torch.tensor(sorted(truth), device=embeddings.device)
    scores = embeddings[users] @ embeddings[data.num_users:].t()
    row_by_user = {int(user): row for row, user in enumerate(users.tolist())}
    for user, item in zip(seen_users.tolist(), seen_items.tolist()):
        if user in row_by_user:
            scores[row_by_user[user], item] = -torch.inf
    ranked = scores.topk(min(k, data.num_items), dim=1).indices.cpu()

    popularity = torch.bincount(data.train_items.cpu(), minlength=data.num_items)
    head_count = max(1, round(data.num_items * head_fraction))
    head_items = set(popularity.topk(head_count).indices.tolist())
    recommended = set(ranked.flatten().tolist())
    head_hits = head_total = tail_hits = tail_total = 0
    for row, user in enumerate(users.tolist()):
        predicted = set(ranked[row].tolist())
        for item in truth[user]:
            if item in head_items:
                head_total += 1
                head_hits += int(item in predicted)
            else:
                tail_total += 1
                tail_hits += int(item in predicted)
    return RankingDiagnostics(
        metrics=ranking_metrics(embeddings, data, split=split, k=k),
        coverage=len(recommended) / data.num_items,
        average_popularity=float(popularity[ranked].float().mean()),
        head_recall=head_hits / head_total if head_total else 0.0,
        tail_recall=tail_hits / tail_total if tail_total else 0.0,
    )


@torch.no_grad()
def ranking_metrics(
    embeddings: torch.Tensor,
    data: RecommendationData,
    *,
    split: str,
    k: int = 20,
) -> RankingMetrics:
    if split == "validation":
        target_users, target_items = data.validation_users, data.validation_items
        seen_users, seen_items = data.train_users, data.train_items
    elif split == "test":
        target_users, target_items = data.test_users, data.test_items
        seen_users = torch.cat((data.train_users, data.validation_users))
        seen_items = torch.cat((data.train_items, data.validation_items))
    else:
        raise ValueError("split must be 'validation' or 'test'")

    truth = _ground_truth(target_users.cpu(), target_items.cpu())
    users = torch.tensor(sorted(truth), device=embeddings.device)
    scores = embeddings[users] @ embeddings[data.num_users:].t()
    row_by_user = {int(user): row for row, user in enumerate(users.tolist())}
    for user, item in zip(seen_users.tolist(), seen_items.tolist()):
        if user in row_by_user:
            scores[row_by_user[user], item] = -torch.inf
    recommendations = scores.topk(min(k, data.num_items), dim=1).indices.cpu()

    recalls, ndcgs = [], []
    for row, user in enumerate(users.tolist()):
        relevant = truth[user]
        ranked = recommendations[row].tolist()
        hits = [int(item in relevant) for item in ranked]
        recalls.append(sum(hits) / len(relevant))
        dcg = sum(hit / torch.log2(torch.tensor(rank + 2.0)).item() for rank, hit in enumerate(hits))
        ideal_hits = min(len(relevant), len(ranked))
        ideal = sum(1 / torch.log2(torch.tensor(rank + 2.0)).item() for rank in range(ideal_hits))
        ndcgs.append(dcg / ideal)
    return RankingMetrics(
        recall=sum(recalls) / len(recalls),
        ndcg=sum(ndcgs) / len(ndcgs),
    )


@torch.no_grad()
def popularity_metrics(data: RecommendationData, *, split: str, k: int = 20) -> RankingMetrics:
    popularity = torch.bincount(data.train_items, minlength=data.num_items).float()
    user_bias = torch.zeros((data.num_users, 1))
    item_scores = popularity[None, :].expand(data.num_users, -1) + user_bias
    # Encode scores as one-hot dimensions so ranking_metrics can reuse its masking and metric logic.
    embeddings = torch.cat((item_scores, torch.eye(data.num_items)), dim=0)
    return ranking_metrics(embeddings, data, split=split, k=k)


def train_recommender(
    model: nn.Module,
    data: RecommendationData,
    *,
    device: torch.device | str = "cpu",
    learning_rate: float = 0.01,
    max_epochs: int = 200,
    patience: int = 20,
    k: int = 20,
    seed: int = 42,
) -> RecommendationResult:
    model = model.to(device)
    device_data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    forbidden = interaction_pairs(data)
    generator = torch.Generator().manual_seed(seed)
    best_ndcg, best_epoch, best_state, stale = -1.0, 0, None, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(data.train_users), generator=generator)
        users = data.train_users[permutation]
        positive = data.train_items[permutation]
        negative = sample_bpr_negatives(
            users, data.num_items, forbidden, generator=generator
        )
        users, positive, negative = users.to(device), positive.to(device), negative.to(device)
        optimizer.zero_grad()
        embeddings = model.get_embedding(device_data.edge_index)
        loss = bpr_loss(embeddings, users, positive, negative, data.num_users)
        loss.backward()
        optimizer.step()
        model.eval()
        embeddings = model.get_embedding(device_data.edge_index)
        validation = ranking_metrics(embeddings, device_data, split="validation", k=k)
        history.append({
            "epoch": float(epoch),
            "loss": float(loss.detach()),
            "val_recall": validation.recall,
            "val_ndcg": validation.ndcg,
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
    embeddings = model.get_embedding(device_data.edge_index)
    return RecommendationResult(
        best_epoch=best_epoch,
        validation=ranking_metrics(embeddings, device_data, split="validation", k=k),
        test=ranking_metrics(embeddings, device_data, split="test", k=k),
        history=history,
    )
