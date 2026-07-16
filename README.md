# STmining: Learning Graph Neural Networks Through Real Tasks

A compact, research-oriented PyTorch project for learning graph neural networks from first
principles. It began with cross-city traffic forecasting and now follows one broader GNN curriculum:
spatiotemporal prediction, static graph tasks, recommendation, knowledge graphs, heterogeneous
graphs, and temporal graphs.

The traffic track compares Historical Average, a node-shared MLP/LSTM, and an STGCN implemented
from first principles. Later lessons use PyTorch Geometric to study GAT, GIN, GPS, graph
autoencoders, LightGCN, R-GCN, HGT, and TGN on public datasets.

## 1. Learning map

- `01–04`: time-series data, baselines, MLP, and LSTM.
- `05–07`: graph message passing, STGCN, and graph diagnostics.
- `08–10`: node, graph, and edge-level GNN tasks.
- `11–15`: recommendation, knowledge graphs, and heterogeneous graphs.
- `16+`: temporal graphs, sequential recommendation, and later scalable/pretrained GNNs.

See [LEARNING_GUIDE.md](LEARNING_GUIDE.md) for the unified roadmap, prerequisites, completed
lessons, experimental conclusions, and planned chapters. See [notebooks/README.md](notebooks/README.md)
for the file-by-file index.

## 2. Installation

Python 3.10–3.12 is supported. Use a fresh environment so an existing system PyTorch installation cannot leak into the project.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
```

For notebooks, register this exact virtual environment as a kernel. Merely activating a
virtual environment before starting a system-wide Jupyter server does not guarantee that
the notebook uses it.

```bash
python -m pip install ipykernel
python -m ipykernel install --user --name stmining --display-name "Python (STmining .venv)"
```

Then select `Python (STmining .venv)` from the notebook kernel menu.

For NVIDIA systems, install the appropriate PyTorch build from the official PyTorch instructions before installing this project.

## 3. Data

The source files follow the layout used by the official DCRNN project:

```text
data/raw/METR-LA/metr-la.h5
data/raw/METR-LA/adj_mx.pkl
data/raw/PEMS-BAY/pems-bay.h5
data/raw/PEMS-BAY/adj_mx_bay.pkl
```

Try the reproducible downloader:

```bash
crosscity download --city all
```

Google Drive occasionally requires an interactive confirmation for large files. If that happens, download the files linked by the [official DCRNN repository](https://github.com/liyaguang/dcrnn), place them in the layout above, and keep the generated archives out of Git. The downloader records the SHA-256 of every retrieved archive; pin those values in your experiment notes before sharing results.

Inspect each dataset before training:

```bash
crosscity inspect --config configs/metr_la.yaml --output artifacts/inspection/metr_la
crosscity inspect --config configs/pems_bay.yaml --output artifacts/inspection/pems_bay
```

## 4. Single-city experiments

All learned models share one command. Start with two epochs as a smoke test, then remove `--max-epochs` for the configured early-stopped run.

```bash
crosscity train --config configs/metr_la.yaml --model ha
crosscity train --config configs/metr_la.yaml --model mlp --max-epochs 2
crosscity train --config configs/metr_la.yaml --model lstm
crosscity train --config configs/metr_la.yaml --model stgcn
```

Repeat for `configs/pems_bay.yaml`. Checkpoints and metrics are written under `artifacts/<city>/<model>/<mode>/seed-<seed>/`.

## 5. Few-shot and transfer experiments

First train the source model on all source-city training data. Then use its checkpoint:

```bash
crosscity transfer \
  --source-config configs/metr_la.yaml \
  --target-config configs/pems_bay.yaml \
  --source-checkpoint artifacts/metr_la/stgcn/target-full/seed-42/best.pt \
  --model stgcn --mode full-finetune --few-shot-days 3 --seed 42
```

Valid modes are `source-zero-shot`, `full-finetune`, and `temporal-frozen`. The scratch comparator is:

```bash
crosscity train --config configs/pems_bay.yaml --model stgcn --few-shot-days 3 --seed 42
```

Run both directions, days `{1,3,7}`, and seeds `{42,43,44}`. Generate aggregate tables with:

```bash
crosscity report --artifacts artifacts --output reports/generated
```

## 6. Evaluation contract

- Inputs: previous 12 five-minute steps; outputs: next 12 steps.
- Chronological split: 70% train, 10% validation, 20% test.
- A sample belongs to a split according to its first target timestamp; target windows never overlap splits.
- Zero, NaN, and infinite speeds are missing and excluded from loss and metrics.
- The scaler uses observed values from the target city's training period only.
- Metrics are speed-unit MAE/RMSE and percentage MAPE at 15, 30, 60 minutes and over all 12 horizons.
- Few-shot conclusions use three seeds and include negative transfer.

## 7. Repository map

```text
configs/                 dataset and experiment defaults
src/crosscity/data/      leakage-safe loading, windows, graph utilities
src/crosscity/models/    HA, MLP, LSTM, and STGCN
src/crosscity/training/  optimization, early stopping, checkpoints
src/crosscity/evaluation masked and horizon-specific metrics
reports/                 report draft and presentation outline
tests/                   data, model, metric, and transfer invariants
```

## Limitations

METR-LA and PEMS-BAY represent two California road-sensor regions rather than a globally diverse collection of cities. This first version measures transfer under controlled feature compatibility; it does not claim geographic universality or state-of-the-art performance. Weather, holidays, POIs, DCRNN, Graph WaveNet, and meta-learning are deliberately out of scope.

## 8. GNN learning notebooks

The notebooks form a cumulative course rather than independent demos. Each lesson introduces a
simple baseline, the relevant mathematics, a leakage-safe evaluation protocol, and a controlled
comparison before using a more expressive model. Current public datasets include Cora, MUTAG,
MovieLens100K, DBLP, and JODIE Wikipedia. The next planned lesson moves to real ecommerce event
sequences with RetailRocket, GRU4Rec, and SASRec.

Notebook code should import reusable implementations from `src/crosscity/`; tests under `tests/`
check data splits, tensor shapes, state updates, and smoke training.
