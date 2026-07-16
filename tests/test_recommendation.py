import torch

from crosscity.data.recommendation import (
    RecommendationData,
    interaction_pairs,
    sample_bpr_negatives,
)
from crosscity.models.recommendation import MatrixFactorization, build_lightgcn
from crosscity.training.recommendation import (
    bpr_loss,
    ranking_diagnostics,
    ranking_metrics,
    train_recommender,
)


def toy_data() -> RecommendationData:
    train_users = torch.tensor([0, 0, 1, 1, 2, 2])
    train_items = torch.tensor([0, 1, 1, 2, 2, 3])
    directed = torch.stack((train_users, train_items + 3))
    return RecommendationData(
        num_users=3, num_items=5, train_users=train_users, train_items=train_items,
        validation_users=torch.tensor([0, 1, 2]), validation_items=torch.tensor([2, 3, 4]),
        test_users=torch.tensor([0, 1, 2]), test_items=torch.tensor([3, 4, 0]),
        edge_index=torch.cat((directed, directed.flip(0)), dim=1),
    )


def test_negative_samples_are_not_observed_interactions():
    data = toy_data()
    negatives = sample_bpr_negatives(
        data.train_users, data.num_items, interaction_pairs(data),
        generator=torch.Generator().manual_seed(3),
    )
    assert all(
        (int(user), int(item)) not in interaction_pairs(data)
        for user, item in zip(data.train_users, negatives)
    )


def test_bpr_prefers_higher_positive_scores():
    embeddings = torch.eye(8)
    embeddings[3] = embeddings[0]
    loss_good = bpr_loss(embeddings, torch.tensor([0]), torch.tensor([0]), torch.tensor([1]), 3)
    loss_bad = bpr_loss(embeddings, torch.tensor([0]), torch.tensor([1]), torch.tensor([0]), 3)
    assert loss_good < loss_bad


def test_models_and_ranking_training_smoke_run():
    data = toy_data()
    for model in (MatrixFactorization(3, 5, 8), build_lightgcn(3, 5, 8, 2)):
        result = train_recommender(model, data, max_epochs=2, patience=2, k=2)
        assert 1 <= result.best_epoch <= 2
        assert 0 <= result.validation.recall <= 1
        assert 0 <= result.test.ndcg <= 1
        embedding = model.get_embedding(data.edge_index)
        assert ranking_metrics(embedding, data, split="test", k=2).recall <= 1


def test_lightgcn_accepts_layer_weights_and_reports_diagnostics():
    data = toy_data()
    alpha = torch.tensor([0.7, 0.2, 0.1])
    model = build_lightgcn(3, 5, 8, 2, alpha=alpha)
    assert torch.allclose(model.alpha, alpha)
    diagnostics = ranking_diagnostics(
        model.get_embedding(data.edge_index), data, split="test", k=2
    )
    assert 0 <= diagnostics.coverage <= 1
    assert diagnostics.average_popularity >= 0
    assert 0 <= diagnostics.head_recall <= 1
    assert 0 <= diagnostics.tail_recall <= 1
