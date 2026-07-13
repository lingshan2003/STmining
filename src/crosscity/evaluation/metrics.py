from __future__ import annotations

import torch


def masked_mae(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = mask.bool()
    if not selected.any():
        return prediction.sum() * 0
    return (prediction[selected] - target[selected]).abs().mean()


def masked_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    selected = mask.bool()
    error = prediction[selected] - target[selected]
    denominator = target[selected].abs().clamp_min(1e-5)
    return {
        "mae": error.abs().mean().item(),
        "rmse": error.square().mean().sqrt().item(),
        "mape": (error.abs() / denominator).mean().mul(100).item(),
    }


def horizon_metrics(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, dict[str, float]]:
    results = {"overall": masked_metrics(prediction, target, mask)}
    for step in (3, 6, 12):
        if prediction.shape[1] >= step:
            results[f"horizon_{step}"] = masked_metrics(
                prediction[:, step - 1], target[:, step - 1], mask[:, step - 1]
            )
    return results

