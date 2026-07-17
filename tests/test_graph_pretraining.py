import torch

from crosscity.data.citation import NodeClassificationData
from crosscity.models.graph_pretraining import (
    GraphEncoder,
    GraphMAE,
    PretrainedNodeClassifier,
    symmetric_contrastive_loss,
)
from crosscity.training.graph_pretraining import (
    few_label_training_data,
    pretrain_contrastive,
    pretrain_graphmae,
)


def toy_data():
    x = torch.eye(8)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    edges = torch.tensor([[0, 1, 1, 2, 4, 5, 5, 6], [1, 0, 2, 1, 5, 4, 6, 5]])
    return NodeClassificationData(
        x=x,
        y=y,
        edge_index=edges,
        train_mask=torch.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=torch.bool),
        val_mask=torch.tensor([0, 0, 1, 0, 0, 0, 1, 0], dtype=torch.bool),
        test_mask=torch.tensor([0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.bool),
    )


def test_few_label_mask_preserves_validation_and_test():
    data = toy_data()
    reduced = few_label_training_data(data, labels_per_class=1, seed=0)
    assert int(reduced.train_mask.sum()) == 2
    assert torch.equal(reduced.val_mask, data.val_mask)
    assert torch.equal(reduced.test_mask, data.test_mask)


def test_graphmae_and_downstream_shapes():
    data = toy_data()
    model = GraphMAE(data.num_features, 8, layers=2, dropout=0)
    mask = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0], dtype=torch.bool)
    reconstruction, representation = model(data.x, data.edge_index, mask)
    assert reconstruction.shape == data.x.shape
    classifier = PretrainedNodeClassifier(model.encoder, 8, data.num_classes)
    assert classifier(data.x, data.edge_index).shape == (data.num_nodes, data.num_classes)


def test_pretraining_losses_run_and_contrastive_positive_pairs_win():
    data = toy_data()
    mae = GraphMAE(data.num_features, 8, dropout=0)
    assert len(pretrain_graphmae(mae, data, epochs=2)) == 2
    encoder = GraphEncoder(data.num_features, 8, dropout=0)
    assert len(pretrain_contrastive(encoder, data, epochs=2, sample_size=4)) == 2
    z = torch.eye(4)
    aligned = symmetric_contrastive_loss(z, z)
    shuffled = symmetric_contrastive_loss(z, z.flip(0))
    assert aligned < shuffled
