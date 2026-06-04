"""Experiment driver for paper-ready synthetic studies."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import random

import torch

from ..config import ExperimentConfig, load_experiment_config
from ..data import generate_episode_split, generate_ltsf_episode_split, get_dataset_profile
from ..models.baselines import FixedWeightPatchTSTModel, correlation_weights, uniform_weights
from ..models.energy_model import TransferEnergyNetwork
from ..models.target_only import TargetOnlyPatchTSTModel
from .evaluation import ExperimentSummary, evaluate_selector, save_experiment_summary
from .trainer import TrainingConfig, compute_forecast_loss, train_step


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
        seq_len=config.data.seq_len,
        source_pooling=config.model.source_pooling,
        forecast_backbone=config.model.forecast_backbone,
        forecast_patch_len=config.model.forecast_patch_len,
        forecast_patch_stride=config.model.forecast_patch_stride,
        forecast_n_heads=config.model.forecast_n_heads,
        forecast_n_layers=config.model.forecast_n_layers,
        forecast_d_ff=config.model.forecast_d_ff,
        forecast_dropout=config.model.forecast_dropout,
        energy_top_k=config.model.energy_top_k,
        energy_logit_clip=config.model.energy_logit_clip,
        energy_center_logits=config.model.energy_center_logits,
    )


def build_target_only_model(
    config: ExperimentConfig,
    target_dim: int,
    horizon: int,
) -> TargetOnlyPatchTSTModel:
    return TargetOnlyPatchTSTModel(
        target_dim=target_dim,
        seq_len=config.data.seq_len,
        horizon=horizon,
        d_model=config.model.hidden_dim,
        patch_len=config.model.forecast_patch_len,
        stride=config.model.forecast_patch_stride,
        n_heads=config.model.forecast_n_heads,
        n_layers=config.model.forecast_n_layers,
        d_ff=config.model.forecast_d_ff,
        dropout=config.model.forecast_dropout,
    )


def get_target_only_config_path(config: ExperimentConfig) -> Path:
    return Path("configs") / f"target_only_{config.data.dataset_name}_patchtst.toml"


def load_target_only_base_state(
    config: ExperimentConfig,
    target_dim: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    target_only_config = load_experiment_config(get_target_only_config_path(config))
    target_only_model = build_target_only_model(
        target_only_config,
        target_dim=target_dim,
        horizon=horizon,
    )
    payload = torch.load(get_best_checkpoint_path(target_only_config), map_location="cpu", weights_only=False)
    target_only_model.load_state_dict(payload["state_dict"])
    return target_only_model.forecaster.state_dict()


def initialize_frozen_base_from_target_only(
    config: ExperimentConfig,
    base_forecaster: torch.nn.Module,
    target_dim: int,
    horizon: int,
) -> None:
    base_forecaster.load_state_dict(load_target_only_base_state(config, target_dim=target_dim, horizon=horizon))
    for parameter in base_forecaster.parameters():
        parameter.requires_grad_(False)


def build_fixed_weight_model(
    config: ExperimentConfig,
    target_dim: int,
    horizon: int,
    selector: str,
) -> FixedWeightPatchTSTModel:
    if selector == "correlation":
        weighting_fn = correlation_weights
    elif selector == "uniform":
        weighting_fn = uniform_weights
    else:
        raise ValueError(f"Unsupported fixed-weight selector: {selector}")

    target_channels = max(1, target_dim // config.data.seq_len) if target_dim % config.data.seq_len == 0 else 1
    return FixedWeightPatchTSTModel(
        seq_len=config.data.seq_len,
        target_channels=target_channels,
        horizon=horizon,
        d_model=config.model.hidden_dim,
        weighting_fn=weighting_fn,
        patch_len=config.model.forecast_patch_len,
        stride=config.model.forecast_patch_stride,
        n_heads=config.model.forecast_n_heads,
        n_layers=config.model.forecast_n_layers,
        d_ff=config.model.forecast_d_ff,
        dropout=config.model.forecast_dropout,
    )


def get_best_checkpoint_path(config: ExperimentConfig) -> Path:
    return Path(config.output_dir) / f"{config.name}_{config.selector}_best.pt"


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


def train_ten_model(
    config: ExperimentConfig,
    save_best_checkpoint: bool = True,
) -> tuple[TransferEnergyNetwork, list, list, list, list[dict], list[dict], int, dict, Path, int]:
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
    initialize_frozen_base_from_target_only(
        config,
        model.forecast_head["base"],
        target_dim=sample_batch.target_context.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
    )
    task_parameters = (
        list(model.target_encoder.parameters())
        + list(model.source_encoder.parameters())
        + list(model.energy_mlp.parameters())
        + list(model.forecast_head["residual"].parameters())
        + [model.source_residual_logit]
    )
    optimizer_task = torch.optim.Adam(task_parameters, lr=config.optim.lr)
    loss_parameters = list(model.future_encoder.parameters()) + list(model.trajectory_energy_mlp.parameters())
    optimizer_loss = torch.optim.Adam(loss_parameters, lr=config.optim.lr_loss)
    train_config = TrainingConfig(
        forecast_loss_weight=config.optim.forecast_loss_weight,
        base_forecast_loss_weight=config.optim.base_forecast_loss_weight,
        forecast_loss_type=config.optim.forecast_loss_type,
        energy_temperature=config.model.energy_temperature,
        energy_margin=config.optim.energy_margin,
        energy_nce_temperature=config.optim.energy_nce_temperature,
        energy_metric_temperature=config.optim.energy_metric_temperature,
        trajectory_energy_weight=config.optim.trajectory_energy_weight,
        trajectory_energy_num_samples=config.optim.trajectory_energy_num_samples,
    )

    train_history: list[dict] = []
    validation_history: list[dict] = []
    best_epoch = 0
    best_val_mse = float("inf")
    best_validation_metrics: dict | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    stopped_epoch = config.optim.epochs
    epochs_without_improvement = 0
    checkpoint_path = get_best_checkpoint_path(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.optim.epochs + 1):
        epoch_losses: list[float] = []
        epoch_forecast: list[float] = []
        epoch_base_forecast: list[float] = []
        epoch_energy: list[float] = []
        epoch_trajectory_energy: list[float] = []

        for batch in train_episodes:
            metrics = train_step(model, optimizer_task, optimizer_loss, batch.as_dict(), train_config)
            epoch_losses.append(float(metrics["loss"].item()))
            epoch_forecast.append(float(metrics["forecast_loss"].item()))
            epoch_base_forecast.append(float(metrics["base_forecast_loss"].item()))
            epoch_energy.append(float(metrics["energy_loss"].item()))
            epoch_trajectory_energy.append(float(metrics["trajectory_energy_loss"].item()))

        train_history.append(
            {
                "epoch": epoch,
                "loss": sum(epoch_losses) / len(epoch_losses),
                "forecast_loss": sum(epoch_forecast) / len(epoch_forecast),
                "base_forecast_loss": sum(epoch_base_forecast) / len(epoch_base_forecast),
                "energy_loss": sum(epoch_energy) / len(epoch_energy),
                "trajectory_energy_loss": sum(epoch_trajectory_energy) / len(epoch_trajectory_energy),
            }
        )

        val_metrics = evaluate_selector("ten", val_episodes, config, model=model)
        val_metrics_dict = {"epoch": epoch, **asdict(val_metrics)}
        validation_history.append(val_metrics_dict)

        if val_metrics.forecast_mse < best_val_mse - config.optim.early_stopping_min_delta:
            best_val_mse = val_metrics.forecast_mse
            best_epoch = epoch
            best_validation_metrics = val_metrics_dict
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            if save_best_checkpoint:
                torch.save(
                    {
                        "state_dict": best_state_dict,
                        "best_epoch": best_epoch,
                        "best_validation_metrics": best_validation_metrics,
                        "config": config.to_dict(),
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.optim.early_stopping_patience:
            stopped_epoch = epoch
            break

    if best_state_dict is None or best_validation_metrics is None:
        raise RuntimeError("Best TEN checkpoint was not captured during training.")

    model.load_state_dict(best_state_dict)
    model.eval()
    return (
        model,
        train_episodes,
        val_episodes,
        test_episodes,
        train_history,
        validation_history,
        best_epoch,
        best_validation_metrics,
        checkpoint_path,
        stopped_epoch,
    )


def load_best_ten_model(
    config: ExperimentConfig,
) -> tuple[TransferEnergyNetwork, list, list, list, dict]:
    train_episodes, val_episodes, test_episodes = load_episode_splits(config)
    sample_batch = train_episodes[0]
    model = build_model(
        config,
        target_dim=sample_batch.target_context.shape[-1],
        source_dim=sample_batch.source_candidates.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
    )
    checkpoint_path = get_best_checkpoint_path(config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, train_episodes, val_episodes, test_episodes, payload


def train_ten_experiment(config: ExperimentConfig) -> ExperimentSummary:
    (
        model,
        _train_episodes,
        _val_episodes,
        test_episodes,
        train_history,
        validation_history,
        best_epoch,
        best_validation_metrics,
        checkpoint_path,
        stopped_epoch,
    ) = train_ten_model(config, save_best_checkpoint=True)
    test_metrics = asdict(evaluate_selector("ten", test_episodes, config, model=model))
    return ExperimentSummary(
        config=config.to_dict(),
        train_history=train_history,
        validation_history=validation_history,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        best_validation_metrics=best_validation_metrics,
        best_checkpoint=str(checkpoint_path),
        stopped_epoch=stopped_epoch,
    )


def train_target_only_experiment(config: ExperimentConfig) -> ExperimentSummary:
    set_seed(config.seed)
    _ = get_dataset_profile(config.data.dataset_name)
    train_episodes, val_episodes, test_episodes = load_episode_splits(config)
    sample_batch = train_episodes[0]
    model = build_target_only_model(
        config,
        target_dim=sample_batch.target_context.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.optim.lr)

    train_history: list[dict] = []
    validation_history: list[dict] = []
    best_epoch = 0
    best_val_mse = float("inf")
    best_validation_metrics: dict | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    stopped_epoch = config.optim.epochs
    epochs_without_improvement = 0
    checkpoint_path = get_best_checkpoint_path(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.optim.epochs + 1):
        epoch_losses: list[float] = []

        for batch in train_episodes:
            model.train()
            optimizer.zero_grad()
            outputs = model(batch.target_context, batch.source_candidates)
            forecast_loss = compute_forecast_loss(
                target=batch.forecast_target,
                mean=outputs["mean"],
                scale=outputs["scale"],
                loss_type=config.optim.forecast_loss_type,
            )
            forecast_loss.backward()
            optimizer.step()
            epoch_losses.append(float(forecast_loss.item()))

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "forecast_loss": avg_loss,
                "transfer_loss": 0.0,
            }
        )

        val_metrics = evaluate_selector("target_only", val_episodes, config, model=model)
        val_metrics_dict = {"epoch": epoch, **asdict(val_metrics)}
        validation_history.append(val_metrics_dict)

        if val_metrics.forecast_mse < best_val_mse - config.optim.early_stopping_min_delta:
            best_val_mse = val_metrics.forecast_mse
            best_epoch = epoch
            best_validation_metrics = val_metrics_dict
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": best_state_dict,
                    "best_epoch": best_epoch,
                    "best_validation_metrics": best_validation_metrics,
                    "config": config.to_dict(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.optim.early_stopping_patience:
            stopped_epoch = epoch
            break

    if best_state_dict is None or best_validation_metrics is None:
        raise RuntimeError("Best target-only checkpoint was not captured during training.")

    model.load_state_dict(best_state_dict)
    model.eval()
    test_metrics = asdict(evaluate_selector("target_only", test_episodes, config, model=model))
    return ExperimentSummary(
        config=config.to_dict(),
        train_history=train_history,
        validation_history=validation_history,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        best_validation_metrics=best_validation_metrics,
        best_checkpoint=str(checkpoint_path),
        stopped_epoch=stopped_epoch,
    )


def train_fixed_weight_experiment(config: ExperimentConfig) -> ExperimentSummary:
    set_seed(config.seed)
    _ = get_dataset_profile(config.data.dataset_name)
    train_episodes, val_episodes, test_episodes = load_episode_splits(config)
    sample_batch = train_episodes[0]
    model = build_fixed_weight_model(
        config,
        target_dim=sample_batch.target_context.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
        selector=config.selector,
    )
    initialize_frozen_base_from_target_only(
        config,
        model.forecast_head["base"],
        target_dim=sample_batch.target_context.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
    )
    optimizer = torch.optim.Adam(
        list(model.forecast_head["residual"].parameters()) + [model.source_residual_logit],
        lr=config.optim.lr,
    )

    train_history: list[dict] = []
    validation_history: list[dict] = []
    best_epoch = 0
    best_val_mse = float("inf")
    best_validation_metrics: dict | None = None
    best_state_dict: dict[str, torch.Tensor] | None = None
    stopped_epoch = config.optim.epochs
    epochs_without_improvement = 0
    checkpoint_path = get_best_checkpoint_path(config)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, config.optim.epochs + 1):
        epoch_losses: list[float] = []

        for batch in train_episodes:
            model.train()
            optimizer.zero_grad()
            outputs = model(
                batch.target_context,
                batch.source_candidates,
                temperature=config.model.energy_temperature,
            )
            forecast_loss = compute_forecast_loss(
                target=batch.forecast_target,
                mean=outputs["mean"],
                scale=outputs["scale"],
                loss_type=config.optim.forecast_loss_type,
            )
            forecast_loss.backward()
            optimizer.step()
            epoch_losses.append(float(forecast_loss.item()))

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        train_history.append(
            {
                "epoch": epoch,
                "loss": avg_loss,
                "forecast_loss": avg_loss,
                "transfer_loss": 0.0,
            }
        )

        val_metrics = evaluate_selector(config.selector, val_episodes, config, model=model)
        val_metrics_dict = {"epoch": epoch, **asdict(val_metrics)}
        validation_history.append(val_metrics_dict)

        if val_metrics.forecast_mse < best_val_mse - config.optim.early_stopping_min_delta:
            best_val_mse = val_metrics.forecast_mse
            best_epoch = epoch
            best_validation_metrics = val_metrics_dict
            best_state_dict = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": best_state_dict,
                    "best_epoch": best_epoch,
                    "best_validation_metrics": best_validation_metrics,
                    "config": config.to_dict(),
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.optim.early_stopping_patience:
            stopped_epoch = epoch
            break

    if best_state_dict is None or best_validation_metrics is None:
        raise RuntimeError(f"Best {config.selector} checkpoint was not captured during training.")

    model.load_state_dict(best_state_dict)
    model.eval()
    test_metrics = asdict(evaluate_selector(config.selector, test_episodes, config, model=model))
    return ExperimentSummary(
        config=config.to_dict(),
        train_history=train_history,
        validation_history=validation_history,
        test_metrics=test_metrics,
        best_epoch=best_epoch,
        best_validation_metrics=best_validation_metrics,
        best_checkpoint=str(checkpoint_path),
        stopped_epoch=stopped_epoch,
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
    elif config.selector == "target_only":
        summary = train_target_only_experiment(config)
    elif config.selector in {"correlation", "uniform"}:
        summary = train_fixed_weight_experiment(config)
    else:
        summary = evaluate_baseline_experiment(config)

    output_dir = Path(config.output_dir)
    output_path = output_dir / f"{config.name}_{config.selector}.json"
    save_experiment_summary(output_path, summary)
    return output_path
