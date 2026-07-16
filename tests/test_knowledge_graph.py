import torch

from crosscity.data.knowledge_graph import (
    make_toy_knowledge_graph,
    sample_corrupted_triples,
)
from crosscity.models.knowledge_graph import DistMult, RGCNDistMult, TransE
from crosscity.training.knowledge_graph import (
    filtered_ranking_metrics,
    knowledge_graph_loss,
    train_knowledge_graph_model,
)


def test_toy_graph_has_inverse_message_edges_without_target_leakage():
    data = make_toy_knowledge_graph()
    assert data.edge_index.shape == (2, 2 * data.train.size(1))
    assert data.edge_type.max() < 2 * data.num_relations
    validation = tuple(map(int, data.validation[:, 0]))
    assert validation not in {
        tuple(map(int, triple)) for triple in data.train.t().tolist()
    }


def test_corrupted_triples_are_not_known_facts():
    data = make_toy_knowledge_graph()
    negative = sample_corrupted_triples(
        data.train,
        data.num_entities,
        data.all_true_triples,
        generator=torch.Generator().manual_seed(7),
    )
    assert all(
        tuple(map(int, triple)) not in data.all_true_triples
        for triple in negative.t().tolist()
    )


def test_transe_and_distmult_scores_have_expected_shape():
    data = make_toy_knowledge_graph()
    for model in (
        TransE(data.num_entities, data.num_relations, 8),
        DistMult(data.num_entities, data.num_relations, 8),
    ):
        entity = model.encode(data.edge_index, data.edge_type)
        score = model.score(entity, data.train)
        assert score.shape == (data.train.size(1),)
        assert knowledge_graph_loss(score, score - 1) > 0


def test_rgcn_training_and_filtered_ranking_smoke_run():
    data = make_toy_knowledge_graph()
    model = RGCNDistMult(data.num_entities, data.num_relations, 8, num_bases=3)
    result = train_knowledge_graph_model(
        model, data, max_epochs=4, patience=4, evaluation_interval=2
    )
    assert 1 <= result.best_epoch <= 4
    assert 0 <= result.validation.mean_reciprocal_rank <= 1
    metrics = filtered_ranking_metrics(model, data, data.test)
    assert metrics.hits_at_1 <= metrics.hits_at_3 <= metrics.hits_at_10
