import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from crosscity.models.graph_classification import GraphGCN, GraphGIN, GraphGPS, GraphMLP
from crosscity.training.graph_classification import train_graph_classifier


def toy_graphs():
    return [
        Data(x=torch.eye(3), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]), y=torch.tensor([0])),
        Data(x=torch.eye(3), edge_index=torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]]), y=torch.tensor([1])),
        Data(x=torch.eye(3), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([0])),
        Data(x=torch.eye(3), edge_index=torch.tensor([[0, 2], [2, 0]]), y=torch.tensor([1])),
    ]


@pytest.mark.parametrize("model_class", [GraphMLP, GraphGCN, GraphGIN, GraphGPS])
def test_graph_models_return_one_prediction_per_graph(model_class):
    batch = Batch.from_data_list(toy_graphs()[:2])
    model = model_class(3, 8, 2)
    logits = model(batch.x, batch.edge_index, batch.batch)
    assert logits.shape == (2, 2)
    logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_graph_training_smoke_run():
    graphs = toy_graphs()
    loader = DataLoader(graphs, batch_size=2, shuffle=False)
    result = train_graph_classifier(
        GraphGIN(3, 8, 2), loader, loader, loader, max_epochs=3, patience=2
    )
    assert 1 <= result.best_epoch <= 3
    assert 0 <= result.validation_accuracy <= 1
    assert 0 <= result.test_accuracy <= 1
