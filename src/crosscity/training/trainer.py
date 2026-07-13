from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from crosscity.config import ExperimentConfig
from crosscity.evaluation.metrics import horizon_metrics, masked_mae


def load_checkpoint(model: torch.nn.Module, path: str | Path, strict: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model_state"], strict=strict)
    return payload


class Trainer:
    def __init__(self, model, adjacency, scaler, config: ExperimentConfig, device: torch.device) -> None:
        self.model = model.to(device)
        self.adjacency = adjacency.to(device) if adjacency is not None else None
        self.scaler = scaler
        self.config = config
        self.device = device
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError("no trainable parameters")
        self.optimizer = torch.optim.Adam(
            parameters, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        amp_enabled = config.amp and device.type == "cuda"
        self.scaler_amp = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled

    def _loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def _move(self, batch):
        return tuple(item.to(self.device) for item in batch)

    def train_epoch(self, dataset) -> float:
        self.model.train()
        total, count = 0.0, 0
        for batch in self._loader(dataset, shuffle=True):
            x, y, mask = self._move(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                prediction = self.model(x, self.adjacency)
                loss = masked_mae(prediction, y, mask)
            self.scaler_amp.scale(loss).backward()
            self.scaler_amp.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler_amp.step(self.optimizer)
            self.scaler_amp.update()
            total += loss.item() * x.shape[0]
            count += x.shape[0]
        return total / max(count, 1)

    @torch.no_grad()
    def predict(self, dataset):
        self.model.eval()
        predictions, targets, masks = [], [], []
        for batch in self._loader(dataset, shuffle=False):
            x, y, mask = self._move(batch)
            predictions.append(self.model(x, self.adjacency).cpu())
            targets.append(y.cpu())
            masks.append(mask.cpu())
        return torch.cat(predictions), torch.cat(targets), torch.cat(masks)

    def evaluate(self, dataset) -> dict[str, dict[str, float]]:
        prediction, target, mask = self.predict(dataset)
        prediction = self.scaler.inverse_transform(prediction)
        target = self.scaler.inverse_transform(target)
        return horizon_metrics(prediction, target, mask)

    def fit(self, train_dataset, val_dataset, checkpoint_path: str | Path) -> list[dict]:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        best_mae, best_state, stale = float("inf"), None, 0
        history = []
        for epoch in range(1, self.config.max_epochs + 1):
            train_loss = self.train_epoch(train_dataset)
            val_metrics = self.evaluate(val_dataset)
            val_mae = val_metrics["overall"]["mae"]
            history.append({"epoch": epoch, "train_loss_scaled": train_loss, "val_mae": val_mae})
            if val_mae < best_mae:
                best_mae, stale = val_mae, 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                stale += 1
            if stale >= self.config.patience:
                break
        if best_state is None:
            raise RuntimeError("training produced no checkpoint")
        self.model.load_state_dict(best_state)
        torch.save(
            {"model_state": best_state, "experiment": asdict(self.config), "best_val_mae": best_mae},
            checkpoint_path,
        )
        return history

