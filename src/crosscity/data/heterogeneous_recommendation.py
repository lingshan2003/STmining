from __future__ import annotations

from dataclasses import dataclass

import torch
from torch_geometric.data import HeteroData


@dataclass(frozen=True)
class HeterogeneousRecommendationData:
    graph: HeteroData
    train_users: torch.Tensor
    train_products: torch.Tensor
    validation_users: torch.Tensor
    validation_products: torch.Tensor
    test_users: torch.Tensor
    test_products: torch.Tensor

    @property
    def num_users(self) -> int:
        return self.graph["user"].num_nodes

    @property
    def num_products(self) -> int:
        return self.graph["product"].num_nodes

    @property
    def known_purchases(self) -> set[tuple[int, int]]:
        users = torch.cat((self.train_users, self.validation_users, self.test_users))
        products = torch.cat((
            self.train_products, self.validation_products, self.test_products
        ))
        return set(zip(users.tolist(), products.tolist()))

    def to(self, device: torch.device | str) -> HeterogeneousRecommendationData:
        return HeterogeneousRecommendationData(
            self.graph.to(device),
            self.train_users.to(device),
            self.train_products.to(device),
            self.validation_users.to(device),
            self.validation_products.to(device),
            self.test_users.to(device),
            self.test_products.to(device),
        )


def _edge(rows: list[tuple[int, int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long).t().contiguous()


def _add_bidirectional(
    graph: HeteroData,
    source: str,
    relation: str,
    target: str,
    edge_index: torch.Tensor,
) -> None:
    graph[source, relation, target].edge_index = edge_index
    graph[target, f"rev_{relation}", source].edge_index = edge_index.flip(0)


def make_toy_ecommerce_graph(
    *, include_behaviors: bool = True, include_metadata: bool = True
) -> HeterogeneousRecommendationData:
    """Build a leakage-safe multi-behaviour product recommendation graph."""
    graph = HeteroData()
    graph["user"].num_nodes = 6
    graph["product"].num_nodes = 12
    graph["category"].num_nodes = 3
    graph["brand"].num_nodes = 3

    train = _edge([
        (0, 0), (0, 1), (1, 1), (1, 2),
        (2, 4), (2, 5), (3, 4), (3, 6),
        (4, 8), (4, 9), (5, 9), (5, 10),
    ])
    validation = _edge([(0, 2), (1, 3), (2, 6), (3, 7), (4, 10), (5, 11)])
    test = _edge([(0, 3), (1, 0), (2, 7), (3, 5), (4, 11), (5, 8)])
    _add_bidirectional(graph, "user", "purchases", "product", train)

    if include_behaviors:
        clicks = _edge([
            (0, 0), (0, 2), (1, 1), (1, 3),
            (2, 4), (2, 6), (3, 5), (3, 7),
            (4, 8), (4, 10), (5, 9), (5, 11),
        ])
        carts = _edge([
            (0, 1), (1, 2), (2, 5), (3, 6), (4, 9), (5, 10),
        ])
        _add_bidirectional(graph, "user", "clicks", "product", clicks)
        _add_bidirectional(graph, "user", "adds_to_cart", "product", carts)

    if include_metadata:
        product_category = _edge([(product, product // 4) for product in range(12)])
        product_brand = _edge([(product, product % 3) for product in range(12)])
        _add_bidirectional(
            graph, "product", "belongs_to", "category", product_category
        )
        _add_bidirectional(graph, "product", "made_by", "brand", product_brand)

    return HeterogeneousRecommendationData(
        graph,
        train[0], train[1],
        validation[0], validation[1],
        test[0], test[1],
    )


def sample_product_negatives(
    users: torch.Tensor,
    num_products: int,
    known: set[tuple[int, int]],
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    negative = torch.randint(num_products, (len(users),), generator=generator)
    for index, user in enumerate(users.tolist()):
        while (user, int(negative[index])) in known:
            negative[index] = torch.randint(num_products, (1,), generator=generator)
    return negative
