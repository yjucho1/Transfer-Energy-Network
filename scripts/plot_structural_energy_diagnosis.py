"""Plot structural energy diagnosis summaries."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot structural energy diagnosis comparison.")
    parser.add_argument("inputs", nargs="+", help="Diagnosis JSON paths")
    parser.add_argument(
        "--output",
        default="results/structural_energy_diagnosis_comparison.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    summaries = [load_summary(Path(path).resolve()) for path in args.inputs]
    datasets = [summary["dataset"] for summary in summaries]
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2"][: len(summaries)]

    mean_gaps = [summary["mean_energy_gap"] for summary in summaries]
    separation_rates = [summary["separation_success_rate"] for summary in summaries]
    margin_rates = [summary["margin_satisfaction_rate"] for summary in summaries]
    mse_values = [summary["mean_sample_mse"] for summary in summaries]
    crps_values = [summary["mean_sample_crps"] for summary in summaries]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()

    axes[0].bar(datasets, mean_gaps, color=colors)
    axes[0].axhline(0.0, color="#999999", linestyle=":", linewidth=1.2)
    axes[0].set_title("Mean Energy Gap")
    axes[0].set_ylabel("E(t,S,y_hat) - E(t,S,y)")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(datasets, separation_rates, color=colors)
    axes[1].axhline(0.5, color="#999999", linestyle=":", linewidth=1.2)
    axes[1].set_title("Separation Success Rate")
    axes[1].set_ylabel("P(E(y) < E(y_hat))")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(datasets, margin_rates, color=colors)
    axes[2].set_title("Margin Satisfaction Rate")
    axes[2].set_ylabel("P(E(y_hat) - E(y) > margin)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(axis="y", alpha=0.25)

    axes[3].scatter(mse_values, mean_gaps, color=colors, s=80)
    for dataset, mse_value, mean_gap in zip(datasets, mse_values, mean_gaps):
        axes[3].annotate(dataset, (mse_value, mean_gap), textcoords="offset points", xytext=(4, 4))
    axes[3].set_title("Forecast Error vs Energy Gap")
    axes[3].set_xlabel("Mean Sample MSE")
    axes[3].set_ylabel("Mean Energy Gap")
    axes[3].grid(alpha=0.25)

    fig.suptitle("Structural Energy Diagnosis", fontsize=14)
    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    print(f"saved_plot={output_path}")


if __name__ == "__main__":
    main()
