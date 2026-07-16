import torch
from torch_geometric.data import Data

from crosscity.data.scalable_node_classification import make_node_masks
from crosscity.models.scalable_gnn import SampledGraphSAGE, ScalableMLP
from crosscity.training import scalable_node_classification as training


def toy_graph() -> Data:
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0],
        [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5],
    ])
    train_mask, valid_mask, test_mask = make_node_masks(
        6, torch.tensor([0, 1]), torch.tensor([2, 3]), torch.tensor([4, 5])
    )
    return Data(
        x=torch.randn(6, 4),
        y=torch.tensor([0, 0, 1, 1, 0, 1]),
        edge_index=edge_index,
        train_mask=train_mask,
        valid_mask=valid_mask,
        test_mask=test_mask,
    )


def test_node_masks_are_disjoint_and_complete_for_given_indices():
    graph = toy_graph()
    assert not (graph.train_mask & graph.valid_mask).any()
    assert not (graph.train_mask & graph.test_mask).any()
    assert int(graph.train_mask.sum() + graph.valid_mask.sum() + graph.test_mask.sum()) == 6


def test_scalable_models_have_expected_output_shapes():
    graph = toy_graph()
    mlp = ScalableMLP(graph.num_features, 8, 2)
    sage = SampledGraphSAGE(graph.num_features, 8, 2, num_layers=2)
    assert mlp(graph.x).shape == (graph.num_nodes, 2)
    assert sage(graph.x, graph.edge_index).shape == (graph.num_nodes, 2)


def test_layerwise_inference_preserves_node_id_order():
    graph = toy_graph()
    batch = graph.clone()
    batch.n_id = torch.arange(graph.num_nodes)
    batch.batch_size = graph.num_nodes
    model = SampledGraphSAGE(graph.num_features, 8, 2, num_layers=2, dropout=0.0)
    model.eval()
    expected = model(graph.x, graph.edge_index)
    actual = model.layerwise_inference(graph.x, [batch], "cpu")
    assert torch.allclose(actual, expected, atol=1e-6)


def test_mlp_and_sampled_training_smoke_run(monkeypatch):
    graph = toy_graph()
    training_batch = graph.clone()
    training_batch.n_id = torch.arange(graph.num_nodes)
    training_batch.batch_size = int(graph.train_mask.sum())
    inference_batch = graph.clone()
    inference_batch.n_id = torch.arange(graph.num_nodes)
    inference_batch.batch_size = graph.num_nodes

    monkeypatch.setattr(
        training,
        "make_neighbor_loaders",
        lambda *args, **kwargs: ([training_batch], [inference_batch]),
    )
    mlp_result = training.train_scalable_mlp(
        ScalableMLP(4, 8, 2), graph, max_epochs=2, patience=2
    )
    sage_result = training.train_sampled_graphsage(
        SampledGraphSAGE(4, 8, 2, num_layers=2),
        graph,
        num_neighbors=[2, 2],
        max_epochs=2,
        patience=2,
    )
    for result in (mlp_result, sage_result):
        assert 1 <= result.best_epoch <= 2
        assert 0 <= result.validation_accuracy <= 1
        assert 0 <= result.test_accuracy <= 1
