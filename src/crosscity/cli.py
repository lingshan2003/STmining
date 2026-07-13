from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from crosscity.config import ProjectConfig, load_config
from crosscity.data import build_data_bundle, load_adjacency, normalize_adjacency
from crosscity.download import DATASETS, download_dataset
from crosscity.evaluation.metrics import horizon_metrics
from crosscity.models import HistoricalAverage, build_model
from crosscity.report import generate_report
from crosscity.training import Trainer, load_checkpoint
from crosscity.utils import choose_device, save_json, seed_everything


def _experiment(config: ProjectConfig, args) -> ProjectConfig:
    updates = {}
    for name in ("model", "seed", "few_shot_days", "max_epochs", "batch_size"):
        value = getattr(args, name, None)
        if value is not None:
            updates[name] = value
    return replace(config, experiment=replace(config.experiment, **updates))


def _components(config: ProjectConfig, few_shot_days: int | None = None):
    bundle = build_data_bundle(config.dataset, few_shot_days=few_shot_days)
    adjacency = normalize_adjacency(load_adjacency(config.dataset.adjacency_path))
    if adjacency.shape[0] != bundle.raw_values.shape[1]:
        raise ValueError("adjacency node count does not match speed matrix")
    model = build_model(
        config.experiment.model, config.dataset.input_steps,
        config.dataset.output_steps, config.experiment.hidden_dim,
    )
    return bundle, adjacency, model


def _artifact_dir(config: ProjectConfig, root: str | Path, source_city: str | None = None) -> Path:
    exp = config.experiment
    parts = [config.dataset.city, exp.model, exp.mode]
    if source_city:
        parts.insert(0, f"{source_city}-to")
    if exp.few_shot_days:
        parts.append(f"{exp.few_shot_days}d")
    parts.append(f"seed-{exp.seed}")
    return Path(root).joinpath(*parts)


def command_download(args) -> None:
    cities = DATASETS if args.city == "all" else [args.city]
    for city in cities:
        print(json.dumps({"city": city, **download_dataset(city, args.destination)}, indent=2))


def command_inspect(args) -> None:
    config = load_config(args.config)
    bundle = build_data_bundle(config.dataset)
    adjacency = load_adjacency(config.dataset.adjacency_path)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    train_end, val_end = bundle.split_points
    summary = {
        "city": config.dataset.city,
        "shape": list(bundle.raw_values.shape),
        "observed_fraction": float(bundle.raw_mask.mean()),
        "split_points": {"train_end": train_end, "val_end": val_end},
        "windows": {"train": len(bundle.train), "val": len(bundle.val), "test": len(bundle.test)},
        "adjacency_density": float((adjacency > 0).mean()),
        "train_mean": bundle.scaler.mean,
        "train_std": bundle.scaler.std,
    }
    save_json(summary, output / "summary.json")
    observed = np.where(bundle.raw_mask, bundle.raw_values, np.nan)
    slots = np.arange(len(observed)) % config.dataset.steps_per_day
    daily = np.array([np.nanmean(observed[slots == slot]) for slot in range(config.dataset.steps_per_day)])
    plt.figure(figsize=(8, 3))
    plt.plot(np.arange(len(daily)) * 5 / 60, daily)
    plt.xlabel("Hour of day")
    plt.ylabel("Mean speed")
    plt.tight_layout()
    plt.savefig(output / "daily_profile.png", dpi=160)
    plt.close()
    print(json.dumps(summary, indent=2))


def _save_run(trainer: Trainer, bundle, run_dir: Path, history=None, metadata=None) -> None:
    metrics = trainer.evaluate(bundle.test)
    save_json({"metadata": metadata or {}, "metrics": metrics}, run_dir / "metrics.json")
    if history is not None:
        save_json(history, run_dir / "history.json")
    print(json.dumps(metrics, indent=2))


def command_train(args) -> None:
    config = _experiment(load_config(args.config), args)
    if config.experiment.few_shot_days is not None:
        config = replace(config, experiment=replace(config.experiment, mode="target-scratch"))
    seed_everything(config.experiment.seed)
    if config.experiment.model == "ha":
        bundle = build_data_bundle(config.dataset)
        train_end, _ = bundle.split_points
        model = HistoricalAverage(config.dataset.steps_per_day).fit(
            bundle.raw_values[:train_end], bundle.raw_mask[:train_end]
        )
        starts = bundle.test.indices
        prediction = torch.from_numpy(model.predict(starts, config.dataset.output_steps)).float()
        target = torch.stack([bundle.test[i][1] for i in range(len(bundle.test))])
        mask = torch.stack([bundle.test[i][2] for i in range(len(bundle.test))])
        target = bundle.scaler.inverse_transform(target)
        metrics = horizon_metrics(prediction, target, mask)
        run_dir = _artifact_dir(config, args.artifacts)
        save_json({"metadata": {"city": config.dataset.city, "model": "ha", "mode": "target-full", "seed": config.experiment.seed}, "metrics": metrics}, run_dir / "metrics.json")
        print(json.dumps(metrics, indent=2))
        return
    bundle, adjacency, model = _components(config, config.experiment.few_shot_days)
    run_dir = _artifact_dir(config, args.artifacts)
    trainer = Trainer(model, adjacency, bundle.scaler, config.experiment, choose_device())
    history = trainer.fit(bundle.train, bundle.val, run_dir / "best.pt")
    metadata = {
        "city": config.dataset.city, "model": config.experiment.model,
        "mode": config.experiment.mode, "few_shot_days": config.experiment.few_shot_days,
        "seed": config.experiment.seed,
    }
    _save_run(trainer, bundle, run_dir, history, metadata)


def command_transfer(args) -> None:
    source = load_config(args.source_config)
    target = _experiment(load_config(args.target_config), args)
    target = replace(target, experiment=replace(target.experiment, mode=args.mode))
    if target.experiment.model not in {"lstm", "stgcn"}:
        raise ValueError("transfer supports lstm and stgcn")
    seed_everything(target.experiment.seed)
    bundle, adjacency, model = _components(target, target.experiment.few_shot_days)
    load_checkpoint(model, args.source_checkpoint)
    run_dir = _artifact_dir(target, args.artifacts, source.dataset.city)
    if args.mode == "source-zero-shot":
        trainer = Trainer(model, adjacency, bundle.scaler, target.experiment, choose_device())
        _save_run(trainer, bundle, run_dir, metadata={
            "city": target.dataset.city, "source_city": source.dataset.city,
            "model": target.experiment.model, "mode": args.mode,
            "few_shot_days": target.experiment.few_shot_days, "seed": target.experiment.seed,
        })
        return
    if args.mode == "temporal-frozen":
        model.freeze_temporal()
    trainer = Trainer(model, adjacency, bundle.scaler, target.experiment, choose_device())
    history = trainer.fit(bundle.train, bundle.val, run_dir / "best.pt")
    _save_run(trainer, bundle, run_dir, history, {
        "city": target.dataset.city, "source_city": source.dataset.city,
        "model": target.experiment.model, "mode": args.mode,
        "few_shot_days": target.experiment.few_shot_days, "seed": target.experiment.seed,
    })


def command_evaluate(args) -> None:
    config = _experiment(load_config(args.config), args)
    bundle, adjacency, model = _components(config)
    load_checkpoint(model, args.checkpoint)
    trainer = Trainer(model, adjacency, bundle.scaler, config.experiment, choose_device())
    metrics = trainer.evaluate(bundle.test)
    if args.output:
        save_json({"metadata": {"city": config.dataset.city, "model": config.experiment.model}, "metrics": metrics}, args.output)
    print(json.dumps(metrics, indent=2))


def command_report(args) -> None:
    csv_path, markdown_path = generate_report(args.artifacts, args.output)
    print(f"wrote {csv_path} and {markdown_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="crosscity", description=__doc__)
    subparsers = parser.add_subparsers(required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--city", choices=[*DATASETS, "all"], default="all")
    download.add_argument("--destination", default="data/raw")
    download.set_defaults(func=command_download)
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--config", required=True)
    inspect.add_argument("--output", default="artifacts/inspection")
    inspect.set_defaults(func=command_inspect)
    for command, function in (("train", command_train), ("evaluate", command_evaluate)):
        item = subparsers.add_parser(command)
        item.add_argument("--config", required=True)
        item.add_argument("--model", choices=["ha", "mlp", "lstm", "stgcn"])
        item.add_argument("--seed", type=int)
        item.add_argument("--few-shot-days", type=int, choices=[1, 3, 7])
        item.add_argument("--batch-size", type=int)
        item.add_argument("--max-epochs", type=int)
        item.add_argument("--artifacts", default="artifacts")
        if command == "evaluate":
            item.add_argument("--checkpoint", required=True)
            item.add_argument("--output")
        item.set_defaults(func=function)
    transfer = subparsers.add_parser("transfer")
    transfer.add_argument("--source-config", required=True)
    transfer.add_argument("--target-config", required=True)
    transfer.add_argument("--source-checkpoint", required=True)
    transfer.add_argument("--model", choices=["lstm", "stgcn"], required=True)
    transfer.add_argument("--mode", choices=["source-zero-shot", "full-finetune", "temporal-frozen"], required=True)
    transfer.add_argument("--few-shot-days", type=int, choices=[1, 3, 7])
    transfer.add_argument("--seed", type=int)
    transfer.add_argument("--batch-size", type=int)
    transfer.add_argument("--max-epochs", type=int)
    transfer.add_argument("--artifacts", default="artifacts")
    transfer.set_defaults(func=command_transfer)
    report = subparsers.add_parser("report")
    report.add_argument("--artifacts", default="artifacts")
    report.add_argument("--output", default="reports/generated")
    report.set_defaults(func=command_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
