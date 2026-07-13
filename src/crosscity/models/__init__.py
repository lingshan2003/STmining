from .baselines import HistoricalAverage, SharedMLP
from .lstm import SharedLSTM
from .stgcn import STGCN


def build_model(name: str, input_steps: int, output_steps: int, hidden_dim: int):
    models = {
        "mlp": lambda: SharedMLP(input_steps, output_steps, hidden_dim),
        "lstm": lambda: SharedLSTM(output_steps, hidden_dim),
        "stgcn": lambda: STGCN(input_steps, output_steps, hidden_dim),
    }
    if name not in models:
        raise ValueError(f"unknown learned model {name!r}; choose from {sorted(models)}")
    return models[name]()


__all__ = ["HistoricalAverage", "SharedMLP", "SharedLSTM", "STGCN", "build_model"]

