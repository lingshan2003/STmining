from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch_geometric.datasets import MovieLens100K


@dataclass(frozen=True)
class RecommendationData:
    num_users: int
    num_items: int
    train_users: torch.Tensor
    train_items: torch.Tensor
    validation_users: torch.Tensor
    validation_items: torch.Tensor
    test_users: torch.Tensor
    test_items: torch.Tensor
    edge_index: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return self.num_users + self.num_items

    def to(self, device: torch.device | str) -> RecommendationData:
        values = {
            name: value.to(device) if isinstance(value, torch.Tensor) else value
            for name, value in self.__dict__.items()
        }
        return RecommendationData(**values)


def load_movielens_implicit(
    root: str | Path,
    *,
    positive_rating: int = 4,
) -> RecommendationData:
    """Load MovieLens100K and create a leakage-safe implicit-feedback graph."""
    graph = MovieLens100K(root=str(Path(root)))[0]
    num_users = graph["user"].num_nodes
    num_items = graph["movie"].num_nodes
    relation = graph["user", "rates", "movie"]

    positive_base = relation.rating >= positive_rating
    base_users = relation.edge_index[0, positive_base]
    base_items = relation.edge_index[1, positive_base]
    base_times = relation.time[positive_base]

    # Hold out each user's latest positive base interaction for validation.
    validation_mask = torch.zeros(len(base_users), dtype=torch.bool)
    for user in base_users.unique():
        positions = torch.where(base_users == user)[0]
        latest = positions[base_times[positions].argmax()]
        validation_mask[latest] = True
    train_mask = ~validation_mask

    test_positive = relation.edge_label >= positive_rating
    test_users = relation.edge_label_index[0, test_positive]
    test_items = relation.edge_label_index[1, test_positive]
    train_users, train_items = base_users[train_mask], base_items[train_mask]

    user_nodes = train_users
    item_nodes = train_items + num_users
    directed = torch.stack((user_nodes, item_nodes))
    edge_index = torch.cat((directed, directed.flip(0)), dim=1)
    return RecommendationData(
        num_users=num_users,
        num_items=num_items,
        train_users=train_users,
        train_items=train_items,
        validation_users=base_users[validation_mask],
        validation_items=base_items[validation_mask],
        test_users=test_users,
        test_items=test_items,
        edge_index=edge_index,
    )


def sample_bpr_negatives(
    users: torch.Tensor,
    num_items: int,
    forbidden_pairs: set[tuple[int, int]],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample one unobserved item per user for pairwise BPR training."""
    negatives = torch.randint(num_items, (len(users),), generator=generator)
    for index, user in enumerate(users.tolist()):
        while (user, int(negatives[index])) in forbidden_pairs:
            negatives[index] = torch.randint(num_items, (1,), generator=generator)
    return negatives


def interaction_pairs(data: RecommendationData) -> set[tuple[int, int]]:
    users = torch.cat((data.train_users, data.validation_users, data.test_users)).tolist()
    items = torch.cat((data.train_items, data.validation_items, data.test_items)).tolist()
    return set(zip(users, items))
