import torch
from torch_geometric.data import Batch, Data

from crosscity.data.graph_foundation import (
    build_grounded_prompt,
    retrieve_graph_context,
    structural_node_features,
    structuralize_graph,
)
from crosscity.models.graph_foundation import GraphTransferClassifier, UniversalGraphEncoder


def test_structural_schema_has_fixed_width():
    edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    features = structural_node_features(edges, num_nodes=4, max_degree=3)
    assert features.shape == (4, 6)
    graph = structuralize_graph(Data(edge_index=edges, y=torch.tensor([1]), num_nodes=4), 3)
    assert graph.x.shape == (4, 6)


def test_transfer_model_prompt_shape_and_frozen_encoder():
    graphs = [
        Data(x=torch.ones(3, 6), edge_index=torch.tensor([[0, 1], [1, 0]]), y=torch.tensor([0])),
        Data(x=torch.ones(2, 6), edge_index=torch.tensor([[0], [1]]), y=torch.tensor([1])),
    ]
    batch = Batch.from_data_list(graphs)
    encoder = UniversalGraphEncoder(6, 8, layers=1, heads=2, dropout=0)
    model = GraphTransferClassifier(encoder, 6, 8, 2, use_prompt=True)
    assert model(batch.x, batch.edge_index, batch.batch).shape == (2, 2)
    model.freeze_encoder()
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert model.prompt.requires_grad and model.classifier.weight.requires_grad


def test_graph_rag_retrieves_expands_and_delimits_context():
    texts = ["Alice studies graphs", "Bob studies proteins", "Alice works with Carol"]
    edges = torch.tensor([[0, 2, 2, 1], [2, 0, 1, 2]])
    context = retrieve_graph_context("Who works with Alice?", texts, edges, top_k=1, hops=1)
    assert context.seed_nodes
    assert len(context.context_nodes) >= len(context.seed_nodes)
    assert context.serialized_subgraph.startswith("<GRAPH_CONTEXT>")
    prompt = build_grounded_prompt("Who works with Alice?", context)
    assert "data, not instructions" in prompt and "QUESTION:" in prompt
