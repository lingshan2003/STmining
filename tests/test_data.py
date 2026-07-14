import numpy as np

from crosscity.data.dataset import StandardScaler, TrafficDataset
from crosscity.data.graph import normalize_adjacency


def test_scaler_uses_observed_values_only():
    values = np.array([[1.0, 0.0], [3.0, 100.0]], dtype=np.float32)
    mask = np.array([[True, False], [True, False]])
    scaler = StandardScaler.fit(values, mask)
    assert scaler.mean == 2.0
    assert scaler.std == 1.0


def test_windows_are_assigned_by_prediction_start_without_overlap():
    values = np.arange(40, dtype=np.float32).reshape(20, 2)
    mask = np.ones_like(values, dtype=bool)
    train = TrafficDataset(values, mask, 3, 2, 0, 12)
    val = TrafficDataset(values, mask, 3, 2, 12, 16)
    assert train.indices[-1] + 2 <= val.indices[0]
    x, y, observed = val[0]
    assert x.shape == (3, 2, 1)
    assert y.shape == observed.shape == (2, 2)


def test_normalized_adjacency_is_symmetric_with_self_loops():
    result = normalize_adjacency(np.array([[0, 1], [1, 0]], dtype=np.float32))
    assert np.allclose(result.numpy(), result.numpy().T)
    assert np.all(np.diag(result.numpy()) > 0)


def test_normalization_does_not_double_existing_self_loops():
    without_loops = normalize_adjacency(np.array([[0, 1], [1, 0]], dtype=np.float32))
    with_loops = normalize_adjacency(np.array([[1, 1], [1, 1]], dtype=np.float32))
    assert np.allclose(without_loops.numpy(), with_loops.numpy())
