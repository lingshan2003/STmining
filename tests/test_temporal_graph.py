import torch
from torch_geometric.data import TemporalData

from crosscity.data.temporal_graph import TemporalGraphSplits
from crosscity.models.temporal_graph import StaticTemporalLinkPredictor, TGNLinkPredictor
from crosscity.training.temporal_graph import (
    temporal_loaders,
    train_static_temporal_baseline,
    train_tgn,
)


def toy_temporal_splits() -> TemporalGraphSplits:
    data = TemporalData(
        src=torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        dst=torch.tensor([2, 3, 3, 2, 2, 3, 3, 2]),
        t=torch.arange(1, 9),
        msg=torch.randn(8, 3),
    )
    return TemporalGraphSplits(data, data[:4], data[4:6], data[6:])


def test_temporal_loader_preserves_event_order_and_samples_destinations():
    splits = toy_temporal_splits()
    train_loader, _, _ = temporal_loaders(splits, batch_size=2)
    batches = list(train_loader)
    assert torch.equal(batches[0].t, torch.tensor([1, 2]))
    assert torch.equal(batches[1].t, torch.tensor([3, 4]))
    assert batches[0].neg_dst.shape == batches[0].dst.shape


def test_static_and_tgn_training_smoke_run():
    splits = toy_temporal_splits()
    static_result = train_static_temporal_baseline(
        StaticTemporalLinkPredictor(splits.full.num_nodes, 8),
        splits,
        batch_size=2,
        max_epochs=1,
    )
    tgn_result = train_tgn(
        TGNLinkPredictor(
            splits.full.num_nodes, splits.full.msg.size(-1),
            memory_dim=8, time_dim=4, embedding_dim=8,
        ),
        splits,
        batch_size=2,
        neighbor_size=2,
        max_epochs=1,
    )
    for result in (static_result, tgn_result):
        assert result.best_epoch == 1
        assert 0 <= result.validation.average_precision <= 1
        assert 0 <= result.test.auc <= 1
