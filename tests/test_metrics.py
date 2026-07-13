import pytest
import torch

from crosscity.evaluation.metrics import masked_metrics


def test_masked_metrics_ignore_missing_targets():
    prediction = torch.tensor([1.0, 1000.0])
    target = torch.tensor([3.0, 2.0])
    mask = torch.tensor([True, False])
    metrics = masked_metrics(prediction, target, mask)
    assert metrics["mae"] == 2.0
    assert metrics["rmse"] == 2.0
    assert metrics["mape"] == pytest.approx(200 / 3)

