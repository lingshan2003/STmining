from pathlib import Path

from crosscity.config import load_config


def test_example_config_loads():
    config = load_config(Path("configs/metr_la.yaml"))
    assert config.dataset.input_steps == config.dataset.output_steps == 12
    assert config.experiment.model == "stgcn"
    assert Path(config.dataset.data_path).is_absolute()
    assert Path(config.dataset.data_path).parent.name == "METR-LA"
