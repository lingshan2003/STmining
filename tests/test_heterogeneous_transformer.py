import torch
from torch_geometric.data import HeteroData

from crosscity.models.heterogeneous_transformer import (
    HeterogeneousMLPClassifier,
    HeterogeneousSAGEClassifier,
    HGTClassifier,
)
from crosscity.training.heterogeneous_node_classification import (
    classification_metrics,
    train_heterogeneous_node_classifier,
)


def toy_dblp() -> HeteroData:
    graph = HeteroData()
    graph["author"].x = torch.randn(8, 6)
    graph["author"].y = torch.tensor([0, 0, 1, 1, 0, 1, 0, 1])
    graph["author"].train_mask = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]).bool()
    graph["author"].val_mask = torch.tensor([0, 0, 0, 0, 1, 1, 0, 0]).bool()
    graph["author"].test_mask = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1]).bool()
    graph["paper"].x = torch.randn(6, 5)
    graph["term"].x = torch.randn(4, 3)
    graph["conference"].num_nodes = 2
    author_paper = torch.tensor([
        [0, 1, 2, 3, 4, 5, 6, 7],
        [0, 0, 1, 1, 2, 3, 4, 5],
    ])
    paper_term = torch.tensor([[0, 1, 2, 3, 4, 5], [0, 0, 1, 2, 3, 3]])
    conference_paper = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 1, 2, 3, 4, 5]])
    graph["author", "writes", "paper"].edge_index = author_paper
    graph["paper", "rev_writes", "author"].edge_index = author_paper.flip(0)
    graph["paper", "has", "term"].edge_index = paper_term
    graph["term", "rev_has", "paper"].edge_index = paper_term.flip(0)
    graph["conference", "publishes", "paper"].edge_index = conference_paper
    graph["paper", "rev_publishes", "conference"].edge_index = conference_paper.flip(0)
    return graph


def graph_schema(graph: HeteroData):
    input_dims = {
        node_type: graph[node_type].x.size(1) if "x" in graph[node_type] else None
        for node_type in graph.node_types
    }
    num_nodes = {
        node_type: graph[node_type].num_nodes for node_type in graph.node_types
    }
    return input_dims, num_nodes


def test_heterogeneous_models_produce_author_logits():
    graph = toy_dblp()
    input_dims, num_nodes = graph_schema(graph)
    models = (
        HeterogeneousMLPClassifier(6, 8, 2),
        HeterogeneousSAGEClassifier(
            input_dims, num_nodes, graph.metadata(), 8, 2, num_layers=1
        ),
        HGTClassifier(
            input_dims, num_nodes, graph.metadata(), 8, 2, num_layers=1, heads=2
        ),
    )
    for model in models:
        assert model(graph).shape == (8, 2)


def test_hgt_training_and_metrics_smoke_run():
    graph = toy_dblp()
    input_dims, num_nodes = graph_schema(graph)
    model = HGTClassifier(
        input_dims, num_nodes, graph.metadata(), 8, 2, num_layers=1, heads=2
    )
    result = train_heterogeneous_node_classifier(
        model, graph, max_epochs=3, patience=3
    )
    assert 1 <= result.best_epoch <= 3
    assert 0 <= result.validation_macro_f1 <= 1
    logits = model(graph)
    accuracy, macro_f1 = classification_metrics(
        logits, graph["author"].y, graph["author"].test_mask, 2
    )
    assert 0 <= accuracy <= 1
    assert 0 <= macro_f1 <= 1
