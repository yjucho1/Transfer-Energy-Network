"""Experiment driver for paper-ready synthetic studies."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random

import torch

from ..config import ExperimentConfig
from ..data import generate_episode_split, generate_ltsf_episode_split, get_dataset_profile
from ..models.energy_model import TransferEnergyNetwork
from .evaluation import ExperimentSummary, evaluate_selector, save_experiment_summary
from .trainer import TrainingConfig, train_step


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def build_model(
    config: ExperimentConfig,
    target_dim: int,
    source_dim: int,
    horizon: int,
) -> TransferEnergyNetwork:
    return TransferEnergyNetwork(
        target_dim=target_dim,
        source_dim=source_dim,
        hidden_dim=config.model.hidden_dim,
        horizon=horizon,
    )


def load_episode_splits(config: ExperimentConfig) -> tuple[list, list, list]:
    if config.data.episode_mode == "real":
        train_episodes = generate_ltsf_episode_split("train", config.data)
        val_episodes = generate_ltsf_episode_split("val", config.data)
        test_episodes = generate_ltsf_episode_split("test", config.data)
        return train_episodes, val_episodes, test_episodes
    train_episodes = generate_episode_split(config.data.train_episodes, config.data)
    val_episodes = generate_episode_split(config.data.val_episodes, config.data)
    test_episodes = generate_episode_split(config.data.test_episodes, config.data)
    return train_episodes, val_episodes, test_episodes


def train_ten_experiment(config: ExperimentConfig) -> ExperimentSummary:
    set_seed(config.seed)
    _ = get_dataset_profile(config.data.dataset_name)
    train_episodes, val_episodes, test_episodes = load_episode_splits(config)
    sample_batch = train_episodes[0]
    model = build_model(
        config,
        target_dim=sample_batch.target_context.shape[-1],
        source_dim=sample_batch.source_candidates.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr)
    train_config = TrainingConfig(
        forecast_loss_weight=config.optim.forecast_loss_weight,
        transfer_loss_weight=config.optim.transfer_loss_weight,
        energy_temperature=config.model.energy_temperature,
        target_temperature=config.optim.target_temperature,
    )

    train_history: list[dict] = []
    validation_history: list[dict] = []

    for epoch in range(1, config.optim.epochs + 1):
        epoch_losses: list[float] = []
        epoch_forecast: list[float] = []
        epoch_transfer: list[float] = []

        for batch in train_episodes:
            metrics = train_step(model, optimizer, batch.as_dict(), train_config)
            epoch_losses.append(float(metrics["loss"].item()))
            epoch_forecast.append(float(metrics["forecast_loss"].item()))
            epoch_transfer.append(float(metrics["transfer_loss"].item()))

        train_history.append(
            {
                "epoch": epoch,
                "loss": sum(epoch_losses) / len(epoch_losses),
                "forecast_loss": sum(epoch_forecast) / len(epoch_forecast),
                "transfer_loss": sum(epoch_transfer) / len(epoch_transfer),
            }
        )

        val_metrics = evaluate_selector("ten", val_episodes, config, model=model)
        validation_history.append({"epoch": epoch, **asdict(val_metrics)})

    test_metrics = asdict(evaluate_selector("ten", test_episodes, config, model=model))
    return ExperimentSummary(
        config=config.to_dict(),
        train_history=train_history,
        validation_history=validation_history,
        test_metrics=test_metrics,
    )


def evaluate_baseline_experiment(config: ExperimentConfig) -> ExperimentSummary:
    set_seed(config.seed)
    _ = get_dataset_profile(config.data.dataset_name)
    train_episodes = []
    _, val_episodes, test_episodes = load_episode_splits(config)
    val_metrics = asdict(evaluate_selector(config.selector, val_episodes, config))
    test_metrics = asdict(evaluate_selector(config.selector, test_episodes, config))

    return ExperimentSummary(
        config=config.to_dict(),
        train_history=train_episodes,
        validation_history=[{"epoch": 0, **val_metrics}],
        test_metrics=test_metrics,
    )


def run_experiment(config: ExperimentConfig) -> Path:
    if config.selector == "ten":
        summary = train_ten_experiment(config)
    else:
        summary = evaluate_baseline_experiment(config)

    output_dir = Path(config.output_dir)
    output_path = output_dir / f"{config.name}_{config.selector}.json"
    save_experiment_summary(output_path, summary)
    return output_path
