import torch

from crosscity.evaluation.metrics import masked_mae
from crosscity.models import SharedMLP


def test_model_can_overfit_one_fixed_batch():
    """Acceptance test for the complete optimizer/backpropagation path."""
    torch.manual_seed(7)
    model = SharedMLP(input_steps=3, output_steps=2, hidden_dim=16)
    x = torch.randn(1, 3, 4, 1)
    # A deterministic target makes this a wiring test, not a generalization test.
    y = torch.stack((x[:, -1, :, 0], x[:, -2, :, 0]), dim=1)
    mask = torch.ones_like(y, dtype=torch.bool)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    initial = masked_mae(model(x), y, mask).item()
    for _ in range(200):
        optimizer.zero_grad()
        loss = masked_mae(model(x), y, mask)
        loss.backward()
        optimizer.step()
    final = masked_mae(model(x), y, mask).item()
    assert final < 0.03
    assert final < initial / 10

