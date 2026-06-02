"""Diagnose structural energy separation for a trained TEN model."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network.config import load_experiment_config
from transfer_energy_network.training.experiment import load_best_ten_model


def gaussian_crps_per_sample(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scale = scale.clamp_min(1e-6)
    z = (target - mean) / scale
    pdf = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return crps.mean(dim=-1)


def pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2 or y.numel() < 2:
        return float("nan")
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.pow(2).sum()) * (y_centered.pow(2).sum())).clamp_min(1e-8)
    return float((x_centered * y_centered).sum().item() / denom.item())


def summarize(config_path: Path) -> dict:
    config = load_experiment_config(config_path)
    model, _, _, test_episodes, _ = load_best_ten_model(config)
    model.eval()

    gaps = []
    successes = []
    margin_hits = []
    mses = []
    crps_scores = []

    with torch.no_grad():
        for batch in test_episodes:
            outputs = model(
                batch.target_context,
                batch.source_candidates,
                temperature=config.model.energy_temperature,
                forecast_target=batch.forecast_target,
            )
            energy_gt = outputs["trajectory_energy_gt"]
            energy_pred = outputs["trajectory_energy_pred"]
            gap = energy_pred - energy_gt
            mse = (outputs["mean"] - batch.forecast_target).pow(2).mean(dim=-1)
            crps = gaussian_crps_per_sample(
                target=batch.forecast_target,
                mean=outputs["mean"],
                scale=outputs["scale"],
            )

            gaps.append(gap.cpu())
            successes.append((energy_gt < energy_pred).float().cpu())
            margin_hits.append((gap > config.optim.energy_margin).float().cpu())
            mses.append(mse.cpu())
            crps_scores.append(crps.cpu())

    gaps_t = torch.cat(gaps)
    successes_t = torch.cat(successes)
    margin_hits_t = torch.cat(margin_hits)
    mse_t = torch.cat(mses)
    crps_t = torch.cat(crps_scores)

    summary = {
        "dataset": config.data.dataset_name,
        "mean_energy_gap": float(gaps_t.mean().item()),
        "median_energy_gap": float(gaps_t.median().item()),
        "separation_success_rate": float(successes_t.mean().item()),
        "margin_satisfaction_rate": float(margin_hits_t.mean().item()),
        "energy_gap_std": float(gaps_t.std(unbiased=False).item()),
        "energy_gap_mse_corr": pearson_corr(gaps_t, mse_t),
        "energy_gap_crps_corr": pearson_corr(gaps_t, crps_t),
        "mean_sample_mse": float(mse_t.mean().item()),
        "mean_sample_crps": float(crps_t.mean().item()),
        "num_test_samples": int(gaps_t.numel()),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose structural energy separation.")
    parser.add_argument("config", help="Path to a TEN config TOML")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    summary = summarize(config_path)

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"saved_diagnosis={output_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
