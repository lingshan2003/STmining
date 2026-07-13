import pytest
import torch

from crosscity.models import SharedLSTM, STGCN
from crosscity.data.graph import normalize_adjacency


@pytest.mark.parametrize("nodes", [5, 8])
@pytest.mark.parametrize("model", [SharedLSTM(output_steps=12, hidden_dim=8), STGCN(12, 12, 8)])
def test_forward_backward_is_node_count_independent(model, nodes):
    x = torch.randn(2, 12, nodes, 1)
    adjacency = normalize_adjacency(torch.eye(nodes))
    prediction = model(x, adjacency)
    assert prediction.shape == (2, 12, nodes)
    prediction.mean().backward()


@pytest.mark.parametrize("factory", [lambda: SharedLSTM(12, 8), lambda: STGCN(12, 12, 8)])
def test_checkpoint_transfers_between_graph_sizes(factory, tmp_path):
    source = factory()
    path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": source.state_dict()}, path)
    target = factory()
    target.load_state_dict(torch.load(path, weights_only=False)["model_state"])
    output = target(torch.randn(1, 12, 7, 1), normalize_adjacency(torch.eye(7)))
    assert output.shape == (1, 12, 7)


def test_temporal_freeze_keeps_prediction_head_trainable():
    model = SharedLSTM(12, 8)
    model.freeze_temporal()
    assert not any(parameter.requires_grad for parameter in model.temporal_encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())

