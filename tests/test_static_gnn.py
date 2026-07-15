import pytest
import torch

from crosscity.data.citation import NodeClassificationData, identity_edges, shuffled_edges
from crosscity.models.static_gnn import GAT, GCN, GraphSAGE, MLPNodeClassifier
from crosscity.training.node_classification import train_node_classifier


def toy_data() -> NodeClassificationData:
    x = torch.eye(6)
    y = torch.tensor([0, 0, 0, 1, 1, 1])
    edge_index = torch.tensor([[0, 1, 1, 2, 3, 4, 4, 5], [1, 0, 2, 1, 4, 3, 5, 4]])
    return NodeClassificationData(
        x=x, y=y, edge_index=edge_index,
        train_mask=torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 1, 0, 0, 1, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 1, 0, 0, 1], dtype=torch.bool),
    )


@pytest.mark.parametrize("model_class", [MLPNodeClassifier, GCN, GraphSAGE, GAT])
def test_node_models_output_shape_and_backward(model_class):
    data = toy_data()
    model = model_class(data.num_features, 8, data.num_classes)
    logits = model(data.x, data.edge_index)
    assert logits.shape == (data.num_nodes, data.num_classes)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_graph_ablations_preserve_valid_node_indices():
    data = toy_data()
    identity = identity_edges(data.num_nodes)
    shuffled = shuffled_edges(data.edge_index, data.num_nodes, seed=7)
    assert identity.shape == (2, data.num_nodes)
    assert shuffled.shape == data.edge_index.shape
    assert int(shuffled.max()) < data.num_nodes
    assert not torch.equal(shuffled, data.edge_index)


def test_toy_training_runs_and_restores_best_epoch():
    torch.manual_seed(0)
    data = toy_data()
    result = train_node_classifier(
        GCN(data.num_features, 8, data.num_classes), data, max_epochs=5, patience=3
    )
    assert 1 <= result.best_epoch <= 5
    assert 0 <= result.validation_accuracy <= 1
    assert 0 <= result.test_accuracy <= 1
