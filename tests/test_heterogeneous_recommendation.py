import torch

from crosscity.data.heterogeneous_recommendation import (
    make_toy_ecommerce_graph,
    sample_product_negatives,
)
from crosscity.models.heterogeneous_recommendation import (
    HeterogeneousEmbedding,
    HeterogeneousGraphSAGE,
)
from crosscity.training.heterogeneous_recommendation import (
    heterogeneous_ranking_metrics,
    train_heterogeneous_recommender,
)


def _num_nodes(data):
    return {
        node_type: data.graph[node_type].num_nodes
        for node_type in data.graph.node_types
    }


def test_toy_graph_has_typed_edges_and_no_purchase_target_leakage():
    data = make_toy_ecommerce_graph()
    assert ("user", "clicks", "product") in data.graph.edge_types
    assert ("product", "belongs_to", "category") in data.graph.edge_types
    graph_purchases = set(map(tuple, data.graph["user", "purchases", "product"].edge_index.t().tolist()))
    assert all(
        (int(user), int(product)) not in graph_purchases
        for user, product in zip(data.validation_users, data.validation_products)
    )


def test_product_negatives_are_not_known_purchases():
    data = make_toy_ecommerce_graph()
    negative = sample_product_negatives(
        data.train_users,
        data.num_products,
        data.known_purchases,
        generator=torch.Generator().manual_seed(5),
    )
    assert all(
        (int(user), int(product)) not in data.known_purchases
        for user, product in zip(data.train_users, negative)
    )


def test_embedding_and_heterogeneous_gnn_training_smoke_run():
    for include_auxiliary in (False, True):
        data = make_toy_ecommerce_graph(
            include_behaviors=include_auxiliary,
            include_metadata=include_auxiliary,
        )
        models = (
            HeterogeneousEmbedding(_num_nodes(data), 8),
            HeterogeneousGraphSAGE(_num_nodes(data), data.graph.metadata(), 8, 1),
        )
        for model in models:
            result = train_heterogeneous_recommender(
                model, data, max_epochs=3, patience=3, k=3
            )
            assert 1 <= result.best_epoch <= 3
            metrics = heterogeneous_ranking_metrics(model, data, split="test", k=3)
            assert 0 <= metrics.recall <= 1
            assert 0 <= metrics.ndcg <= 1
