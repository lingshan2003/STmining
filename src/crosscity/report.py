from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def collect_results(artifacts: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(artifacts).glob("**/metrics.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {})
        for horizon, metrics in payload["metrics"].items():
            rows.append({**metadata, "horizon": horizon, **metrics, "path": str(path)})
    return pd.DataFrame(rows)


def generate_report(artifacts: str | Path, output: str | Path) -> tuple[Path, Path]:
    frame = collect_results(artifacts)
    if frame.empty:
        raise ValueError(f"no metrics.json files under {artifacts}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "results.csv"
    frame.to_csv(csv_path, index=False)
    group_columns = [c for c in ("city", "source_city", "model", "mode", "few_shot_days", "horizon") if c in frame]
    summary = frame.groupby(group_columns, dropna=False)[["mae", "rmse", "mape"]].agg(["mean", "std"])
    markdown_path = output / "results.md"
    markdown_path.write_text(
        "# Experiment Results\n\n" + summary.to_markdown() + "\n\n"
        "> Values aggregate random seeds. `std` is undefined when only one seed exists.\n",
        encoding="utf-8",
    )
    return csv_path, markdown_path
