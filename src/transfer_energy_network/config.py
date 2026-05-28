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
    episode_mode: str = "real"
    task_name: str = "long_term_forecast"
    features: str = "M"
    target: str = "OT"
    freq: str = "h"
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 336
    adaptation_backbone: str = "ridge"
    adaptation_lag: int = 24
    adaptation_ridge: float = 1e-2
    adaptation_val_ratio: float = 0.25
    adaptation_min_samples: int = 8
    adaptation_patch_len: int = 8
    adaptation_patch_stride: int = 4
    adaptation_d_model: int = 16
    adaptation_n_heads: int = 2
    adaptation_n_layers: int = 1
    adaptation_d_ff: int = 32
    adaptation_dropout: float = 0.1
    adaptation_epochs: int = 2
    adaptation_lr: float = 1e-3
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
    energy_temperature: float = 0.7


@dataclass
class OptimConfig:
    lr: float = 1e-3
    epochs: int = 5
    forecast_loss_weight: float = 1.0
    transfer_loss_weight: float = 0.5
    target_temperature: float = 0.7


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
