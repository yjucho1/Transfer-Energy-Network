"""Render one sample forecast comparison plot."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network.config import load_experiment_config
from transfer_energy_network.data import load_ltsf_series
from transfer_energy_network.models.baselines import oracle_weights
from transfer_energy_network.training.losses import gaussian_quantile
from transfer_energy_network.training.evaluation import build_forecast_from_weights
from transfer_energy_network.training.experiment import (
    build_fixed_weight_model,
    build_target_only_model,
    get_best_checkpoint_path,
    load_best_ten_model,
    train_ten_model as fit_best_ten_model,
    load_episode_splits,
)


def load_or_train_best_ten_model(config):
    checkpoint_path = get_best_checkpoint_path(config)
    if checkpoint_path.exists():
        model, _, _, test_episodes, _ = load_best_ten_model(config)
        return model, test_episodes

    model, _, _, test_episodes, *_ = fit_best_ten_model(
        config,
        save_best_checkpoint=True,
    )
    return model, test_episodes


def load_best_target_only_model(
    config,
    config_path: Path,
    target_only_config_override: str | None = None,
):
    if target_only_config_override is None:
        target_only_config_path = config_path.with_name(
            f"target_only_{config.data.dataset_name}_patchtst.toml"
        )
    else:
        target_only_config_path = Path(target_only_config_override).resolve()
    target_only_config = load_experiment_config(target_only_config_path)
    checkpoint_path = get_best_checkpoint_path(target_only_config)

    train_episodes, _, test_episodes = load_episode_splits(target_only_config)
    sample_batch = train_episodes[0]
    model = build_target_only_model(
        target_only_config,
        target_dim=sample_batch.target_context.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, test_episodes


def load_best_fixed_weight_model(config, config_path: Path, selector: str):
    baseline_config_path = config_path.with_name(
        f"{selector}_{config.data.dataset_name}_patchtst.toml"
    )
    baseline_config = load_experiment_config(baseline_config_path)
    checkpoint_path = get_best_checkpoint_path(baseline_config)

    train_episodes, _, test_episodes = load_episode_splits(baseline_config)
    sample_batch = train_episodes[0]
    model = build_fixed_weight_model(
        baseline_config,
        target_dim=sample_batch.target_context.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
        selector=selector,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, test_episodes


def get_target_denormalization_stats(config):
    loaded = load_ltsf_series(config.data)
    train_end = int(loaded.values.shape[0] * config.data.train_ratio)
    train_values = loaded.values[:train_end]
    target_mean = train_values[:, loaded.target_index].mean()
    target_std = train_values[:, loaded.target_index].std().clamp_min(1e-5)
    return target_mean, target_std


def denormalize_target(values: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor) -> torch.Tensor:
    return values * target_std + target_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one forecast comparison sample.")
    parser.add_argument("config", help="Path to TOML config")
    parser.add_argument(
        "--target-only-config",
        default=None,
        help="Optional target-only TOML config to overlay in the plot.",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/forecast_comparison_etth1_sample0.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_experiment_config(config_path)
    model, test_episodes = load_or_train_best_ten_model(config)
    target_only_model, target_only_test_episodes = load_best_target_only_model(
        config,
        config_path,
        args.target_only_config,
    )
    correlation_model, correlation_test_episodes = load_best_fixed_weight_model(
        config,
        config_path,
        "correlation",
    )
    uniform_model, uniform_test_episodes = load_best_fixed_weight_model(
        config,
        config_path,
        "uniform",
    )
    target_mean, target_std = get_target_denormalization_stats(config)
    batch = test_episodes[0]
    target_only_batch = target_only_test_episodes[0]
    correlation_batch = correlation_test_episodes[0]
    uniform_batch = uniform_test_episodes[0]
    sample_idx = args.sample_index
    loaded = load_ltsf_series(config.data)
    seq_len = config.data.seq_len
    num_features = batch.target_context.shape[-1] // seq_len
    reshaped_context = batch.target_context[sample_idx].reshape(seq_len, num_features).cpu()
    target_history = reshaped_context[:, loaded.target_index]

    with torch.no_grad():
        ten_outputs = model(
            batch.target_context,
            batch.source_candidates,
            temperature=config.model.energy_temperature,
        )
        ten_pred = ten_outputs["mean"][sample_idx].cpu()
        ten_scale = ten_outputs["scale"][sample_idx].cpu()
        ten_q10 = gaussian_quantile(ten_pred, ten_scale, 0.1)
        ten_q90 = gaussian_quantile(ten_pred, ten_scale, 0.9)
        target_only_outputs = target_only_model(
            target_only_batch.target_context,
            target_only_batch.source_candidates,
            temperature=config.model.energy_temperature,
        )
        target_only_pred = target_only_outputs["mean"][sample_idx].cpu()
        correlation_outputs = correlation_model(
            correlation_batch.target_context,
            correlation_batch.source_candidates,
            temperature=config.model.energy_temperature,
        )
        correlation_pred = correlation_outputs["mean"][sample_idx].cpu()
        uniform_outputs = uniform_model(
            uniform_batch.target_context,
            uniform_batch.source_candidates,
            temperature=config.model.energy_temperature,
        )
        uniform_pred = uniform_outputs["mean"][sample_idx].cpu()
    oracle_pred, _ = build_forecast_from_weights(
        batch,
        oracle_weights(batch.validation_delta),
        config,
    )

    actual = batch.forecast_target[sample_idx].cpu()
    target_history = denormalize_target(target_history, target_mean, target_std)
    actual = denormalize_target(actual, target_mean, target_std)
    ten_pred = denormalize_target(ten_pred, target_mean, target_std)
    ten_q10 = denormalize_target(ten_q10, target_mean, target_std)
    ten_q90 = denormalize_target(ten_q90, target_mean, target_std)
    target_only_pred = denormalize_target(target_only_pred, target_mean, target_std)
    correlation_pred = denormalize_target(correlation_pred, target_mean, target_std)
    uniform_pred = denormalize_target(uniform_pred, target_mean, target_std)
    oracle_pred = denormalize_target(oracle_pred[sample_idx].cpu(), target_mean, target_std)

    future_x = torch.arange(seq_len, seq_len + actual.shape[0]).cpu()
    history_x = torch.arange(seq_len).cpu()

    plt.figure(figsize=(13, 6.5))
    plt.plot(history_x, target_history, label="Context History", color="#7F7F7F", linewidth=2.2)
    plt.axvline(seq_len - 1, color="#BBBBBB", linestyle=":", linewidth=1.4)
    plt.plot(future_x, actual, label="Actual Future", color="#111111", linewidth=2.4)
    plt.fill_between(
        future_x.numpy(),
        ten_q10.numpy(),
        ten_q90.numpy(),
        color="#C44E52",
        alpha=0.18,
        label="TEN P10-P90",
        zorder=3,
    )
    plt.plot(future_x, ten_pred, label="TEN", color="#C44E52", linewidth=2.0, zorder=4)
    plt.plot(
        future_x,
        target_only_pred,
        label="Target-only PatchTST",
        color="#DD8452",
        linewidth=2.0,
        linestyle="-.",
        zorder=4,
    )
    plt.plot(
        future_x,
        correlation_pred,
        label="Correlation",
        color="#4C72B0",
        linewidth=1.8,
        zorder=2,
    )
    plt.plot(
        future_x,
        uniform_pred,
        label="Uniform",
        color="#55A868",
        linewidth=1.8,
        zorder=2,
    )
    plt.plot(
        future_x,
        oracle_pred,
        label="Oracle",
        color="#8172B2",
        linewidth=2.2,
        linestyle="--",
        marker="o",
        markersize=3,
        markevery=max(1, actual.shape[0] // 20),
        zorder=5,
    )
    plt.title(f"Forecast Comparison on {config.data.dataset_name} Test Sample {sample_idx}")
    plt.xlabel("Time Step")
    plt.ylabel(f"{config.data.target} (Original Scale)")
    plt.legend(frameon=False, ncol=3)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    print(f"saved_plot={output_path}")


if __name__ == "__main__":
    main()
