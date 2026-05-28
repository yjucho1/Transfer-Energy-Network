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
from transfer_energy_network.models.baselines import correlation_weights, oracle_weights, uniform_weights
from transfer_energy_network.training.losses import gaussian_quantile
from transfer_energy_network.training.evaluation import build_forecast_from_weights
from transfer_energy_network.training.experiment import build_model, load_episode_splits, set_seed
from transfer_energy_network.training.trainer import TrainingConfig, train_step


def train_ten_model(config):
    set_seed(config.seed)
    train_episodes, _, test_episodes = load_episode_splits(config)
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

    for _ in range(config.optim.epochs):
        for batch in train_episodes:
            train_step(model, optimizer, batch.as_dict(), train_config)
    return model.eval(), test_episodes


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot one forecast comparison sample.")
    parser.add_argument("config", help="Path to TOML config")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--output",
        default="results/forecast_comparison_etth1_sample0.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    model, test_episodes = train_ten_model(config)
    batch = test_episodes[0]
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

    corr_weights = correlation_weights(
        batch.target_context,
        batch.source_candidates,
        temperature=config.model.energy_temperature,
    )
    corr_pred, _ = build_forecast_from_weights(batch, corr_weights, config)

    uniform_pred, _ = build_forecast_from_weights(
        batch,
        uniform_weights(batch.target_context, batch.source_candidates),
        config,
    )
    oracle_pred, _ = build_forecast_from_weights(
        batch,
        oracle_weights(batch.validation_delta),
        config,
    )

    actual = batch.forecast_target[sample_idx].cpu()
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
        corr_pred[sample_idx].cpu(),
        label="Correlation",
        color="#4C72B0",
        linewidth=1.8,
        zorder=2,
    )
    plt.plot(
        future_x,
        uniform_pred[sample_idx].cpu(),
        label="Uniform",
        color="#55A868",
        linewidth=1.8,
        zorder=2,
    )
    plt.plot(
        future_x,
        oracle_pred[sample_idx].cpu(),
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
    plt.ylabel("Normalized Target Value")
    plt.legend(frameon=False, ncol=3)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    print(f"saved_plot={output_path}")


if __name__ == "__main__":
    main()
