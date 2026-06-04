"""Configuration schema for repeatable research experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    import tomli as tomllib


@dataclass
class DataConfig:
    dataset_bundle: str = "./datasets"
    dataset_name: str = "etth1"
    file_name: str = "ETT-small/ETTh1.csv"
    source_pool_datasets: list[str] = field(default_factory=list)
    episode_mode: str = "real"
    task_name: str = "long_term_forecast"
    features: str = "M"
    freq: str = "h"
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 336
    train_ratio: float = 0.7
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    window_stride: int = 8
    max_windows_per_split: int = 256
    train_episodes: int = 100
    val_episodes: int = 30
    test_episodes: int = 30
    batch_size: int = 16
    num_sources: int = 6
    target_dim: int = 10
    source_dim: int = 10
    horizon: int = 4
    helpful_delta: float = 1.5
    harmful_delta: float = -0.5
    source_signal_scale: float = 0.4
    target_signal_scale: float = 0.6
    noise_scale: float = 0.1


@dataclass
class ModelConfig:
    hidden_dim: int = 24
    source_pooling: str = "energy"
    energy_temperature: float = 0.7
    energy_top_k: int = 0
    energy_logit_clip: float = 0.0
    energy_center_logits: bool = True
    forecast_backbone: str = "patchtst"
    forecast_patch_len: int = 8
    forecast_patch_stride: int = 4
    forecast_n_heads: int = 2
    forecast_n_layers: int = 2
    forecast_d_ff: int = 64
    forecast_dropout: float = 0.1


@dataclass
class OptimConfig:
    lr: float = 1e-3
    lr_loss: float = 1e-4
    epochs: int = 5
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 1e-4
    forecast_loss_weight: float = 1.0
    base_forecast_loss_weight: float = 1.0
    forecast_loss_type: str = "crps"
    energy_margin: float = 0.2
    energy_nce_temperature: float = 1.0
    energy_metric_temperature: float = 0.1
    trajectory_energy_weight: float = 0.1
    trajectory_energy_num_samples: int = 4


@dataclass
class ExperimentConfig:
    name: str = "synthetic_ten"
    seed: int = 21
    selector: str = "ten"
    output_dir: str = "results"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge_dataclass(cls, values: dict | None):
    if values is None:
        return cls()
    return cls(**values)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return ExperimentConfig(
        name=raw.get("name", "synthetic_ten"),
        seed=raw.get("seed", 21),
        selector=raw.get("selector", "ten"),
        output_dir=raw.get("output_dir", "results"),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        optim=_merge_dataclass(OptimConfig, raw.get("optim")),
    )
