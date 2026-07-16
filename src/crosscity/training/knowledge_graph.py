from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from crosscity.data.knowledge_graph import KnowledgeGraphData, sample_corrupted_triples


@dataclass(frozen=True)
class KnowledgeGraphMetrics:
    mean_reciprocal_rank: float
    hits_at_1: float
    hits_at_3: float
    hits_at_10: float


@dataclass(frozen=True)
class KnowledgeGraphResult:
    best_epoch: int
    validation: KnowledgeGraphMetrics
    test: KnowledgeGraphMetrics
    history: list[dict[str, float]]


def knowledge_graph_loss(positive_score: torch.Tensor, negative_score: torch.Tensor) -> torch.Tensor:
    """Logistic loss that raises true scores and lowers corrupted scores."""
    return F.softplus(-positive_score).mean() + F.softplus(negative_score).mean()


@torch.no_grad()
def filtered_ranking_metrics(
    model: nn.Module,
    data: KnowledgeGraphData,
    triples: torch.Tensor,
) -> KnowledgeGraphMetrics:
    """Rank the true head and tail while filtering every other known true fact."""
    model.eval()
    entity = model.encode(data.edge_index, data.edge_type)
    known = data.all_true_triples
    ranks: list[int] = []
    candidates = torch.arange(data.num_entities, device=entity.device)
    for head, relation, tail in triples.t().tolist():
        tail_triples = torch.stack((
            torch.full_like(candidates, head),
            torch.full_like(candidates, relation),
            candidates,
        ))
        tail_scores = model.score(entity, tail_triples)
        for candidate in range(data.num_entities):
            if candidate != tail and (head, relation, candidate) in known:
                tail_scores[candidate] = -torch.inf
        ranks.append(1 + int((tail_scores > tail_scores[tail]).sum()))

        head_triples = torch.stack((
            candidates,
            torch.full_like(candidates, relation),
            torch.full_like(candidates, tail),
        ))
        head_scores = model.score(entity, head_triples)
        for candidate in range(data.num_entities):
            if candidate != head and (candidate, relation, tail) in known:
                head_scores[candidate] = -torch.inf
        ranks.append(1 + int((head_scores > head_scores[head]).sum()))

    rank = torch.tensor(ranks, dtype=torch.float)
    return KnowledgeGraphMetrics(
        mean_reciprocal_rank=float((1 / rank).mean()),
        hits_at_1=float((rank <= 1).float().mean()),
        hits_at_3=float((rank <= 3).float().mean()),
        hits_at_10=float((rank <= 10).float().mean()),
    )


def train_knowledge_graph_model(
    model: nn.Module,
    data: KnowledgeGraphData,
    *,
    device: torch.device | str = "cpu",
    learning_rate: float = 0.01,
    max_epochs: int = 500,
    patience: int = 80,
    evaluation_interval: int = 5,
    seed: int = 42,
) -> KnowledgeGraphResult:
    model = model.to(device)
    device_data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    known = data.all_true_triples
    best_mrr, best_epoch, best_state, stale = -1.0, 0, None, 0
    history: list[dict[str, float]] = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        negative = sample_corrupted_triples(
            data.train, data.num_entities, known, generator=generator
        ).to(device)
        optimizer.zero_grad()
        entity = model.encode(device_data.edge_index, device_data.edge_type)
        loss = knowledge_graph_loss(
            model.score(entity, device_data.train), model.score(entity, negative)
        )
        loss.backward()
        optimizer.step()
        if epoch % evaluation_interval != 0 and epoch != max_epochs:
            continue
        validation = filtered_ranking_metrics(
            model, device_data, device_data.validation
        )
        history.append({
            "epoch": float(epoch),
            "loss": float(loss.detach()),
            "validation_mrr": validation.mean_reciprocal_rank,
        })
        if validation.mean_reciprocal_rank > best_mrr:
            best_mrr, best_epoch = validation.mean_reciprocal_rank, epoch
            best_state, stale = deepcopy(model.state_dict()), 0
        else:
            stale += evaluation_interval
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return KnowledgeGraphResult(
        best_epoch,
        filtered_ranking_metrics(model, device_data, device_data.validation),
        filtered_ranking_metrics(model, device_data, device_data.test),
        history,
    )
