import torch

from crosscity.data.citation import NodeClassificationData
from crosscity.data.link_prediction import make_link_prediction_splits
from crosscity.models.link_prediction import build_gae, build_vgae
from crosscity.training.link_prediction import train_link_predictor


def toy_graph() -> NodeClassificationData:
    edges = [(i, i + 1) for i in range(9)] + [(0, 4), (2, 7), (3, 8)]
    directed = edges + [(target, source) for source, target in edges]
    edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
    return NodeClassificationData(
        x=torch.eye(10), y=torch.arange(10) % 2, edge_index=edge_index,
        train_mask=torch.ones(10, dtype=torch.bool),
        val_mask=torch.zeros(10, dtype=torch.bool),
        test_mask=torch.zeros(10, dtype=torch.bool),
    )


def test_link_split_separates_message_edges_and_labels():
    splits = make_link_prediction_splits(
        toy_graph(), validation_fraction=0.2, test_fraction=0.2, seed=7
    )
    assert splits.train.pos_edge_label_index.shape[1] > 0
    assert splits.train.neg_edge_label_index.shape == splits.train.pos_edge_label_index.shape
    assert splits.validation.pos_edge_label_index.shape == splits.validation.neg_edge_label_index.shape
    assert splits.test.pos_edge_label_index.shape == splits.test.neg_edge_label_index.shape
    assert splits.train.edge_index.shape[1] < toy_graph().edge_index.shape[1]


def test_gae_and_vgae_training_smoke_run():
    splits = make_link_prediction_splits(
        toy_graph(), validation_fraction=0.2, test_fraction=0.2, seed=7
    )
    for model in (build_gae(10, 8, 4), build_vgae(10, 8, 4)):
        result = train_link_predictor(
            model, splits.train, splits.validation, splits.test, max_epochs=2, patience=2
        )
        assert 1 <= result.best_epoch <= 2
        assert 0 <= result.validation.auc <= 1
        assert 0 <= result.test.average_precision <= 1
