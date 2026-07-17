import torch

from crosscity.data.gnn_failure_modes import (
    edge_homophily,
    feature_knn_edges,
    make_tree_retrieval_benchmark,
)
from crosscity.models.gnn_failure_modes import DeepGCN, propagation_trace
from crosscity.training.node_classification import train_node_classifier


def test_tree_retrieval_is_leakage_safe_and_rewiring_only_changes_edges():
    benchmark = make_tree_retrieval_benchmark(num_trees=10, depth=3, seed=7)
    data = benchmark.data
    assert data.num_nodes == 10 * (2**4 - 1)
    assert int(data.train_mask.sum() + data.val_mask.sum() + data.test_mask.sum()) == 10
    assert not torch.any(data.train_mask & data.val_mask)
    supervised = data.train_mask | data.val_mask | data.test_mask
    expected = torch.isin(torch.arange(data.num_nodes), benchmark.roots)
    assert torch.equal(supervised, expected)
    rewired = benchmark.with_root_shortcuts()
    assert rewired.edge_index.shape[1] > data.edge_index.shape[1]
    assert torch.equal(rewired.x, data.x)
    assert torch.equal(rewired.y, data.y)
    assert torch.equal(rewired.train_mask, data.train_mask)


def test_feature_knn_is_valid_and_label_free():
    x = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    edges = feature_knn_edges(x, k=1, chunk_size=2)
    assert edges.shape[0] == 2
    assert int(edges.min()) >= 0 and int(edges.max()) < len(x)
    assert not torch.any(edges[0] == edges[1])
    labels = torch.tensor([0, 0, 1, 1])
    assert edge_homophily(edges, labels) == 1.0


def test_propagation_trace_detects_smoothing():
    x = torch.eye(4)
    edges = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]])
    trace = propagation_trace(x, edges, steps=8)
    assert len(trace) == 9
    assert trace[-1]["feature_variance"] < trace[0]["feature_variance"]
    assert trace[-1]["dirichlet_energy"] < trace[0]["dirichlet_energy"]
    assert trace[-1]["mean_pairwise_cosine"] > trace[0]["mean_pairwise_cosine"]


def test_deep_gcn_shape_backward_and_toy_training():
    benchmark = make_tree_retrieval_benchmark(num_trees=10, depth=2, seed=3)
    data = benchmark.data
    model = DeepGCN(3, 8, 2, num_layers=2, residual=True, normalization="layer", dropout=0)
    logits = model(data.x, data.edge_index)
    assert logits.shape == (data.num_nodes, 2)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    result = train_node_classifier(model, data, max_epochs=2, patience=2)
    assert 1 <= result.best_epoch <= 2
