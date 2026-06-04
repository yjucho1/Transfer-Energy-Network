"""Diagnose source contribution via Delta-E for a trained TEN model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network.config import load_experiment_config
from transfer_energy_network.training.experiment import load_best_ten_model


def summarize(config_path: Path) -> dict:
    config = load_experiment_config(config_path)
    model, _, _, test_episodes, _ = load_best_ten_model(config)
    model.eval()

    delta_gt_values = []
    delta_pred_values = []
    target_summary: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"gt": [], "pred": []})

    with torch.no_grad():
        for batch in test_episodes:
            outputs = model(
                batch.target_context,
                batch.source_candidates,
                temperature=config.model.energy_temperature,
                forecast_target=batch.forecast_target,
            )

            delta_gt = (outputs["correction_energy_gt"] - outputs["correction_energy_base"]).cpu()
            delta_pred = (outputs["correction_energy_pred"] - outputs["correction_energy_base"]).cpu()
            delta_gt_values.append(delta_gt)
            delta_pred_values.append(delta_pred)

            target_labels = batch.metadata.get("target_labels", []) if batch.metadata else []
            for idx, label in enumerate(target_labels):
                target_summary[label]["gt"].append(float(delta_gt[idx].item()))
                target_summary[label]["pred"].append(float(delta_pred[idx].item()))

    delta_gt_t = torch.cat(delta_gt_values)
    delta_pred_t = torch.cat(delta_pred_values)

    per_target = {}
    for target_label, values in sorted(target_summary.items()):
        gt_mean = sum(values["gt"]) / max(1, len(values["gt"]))
        pred_mean = sum(values["pred"]) / max(1, len(values["pred"]))
        per_target[target_label] = {
            "mean_delta_e_gt": gt_mean,
            "mean_delta_e_pred": pred_mean,
            "num_samples": len(values["gt"]),
        }

    return {
        "dataset": config.data.dataset_name,
        "mean_delta_e_gt": float(delta_gt_t.mean().item()),
        "mean_delta_e_pred": float(delta_pred_t.mean().item()),
        "negative_delta_e_gt_rate": float((delta_gt_t < 0).float().mean().item()),
        "negative_delta_e_pred_rate": float((delta_pred_t < 0).float().mean().item()),
        "delta_e_gt_std": float(delta_gt_t.std(unbiased=False).item()),
        "delta_e_pred_std": float(delta_pred_t.std(unbiased=False).item()),
        "num_test_samples": int(delta_gt_t.numel()),
        "per_target": per_target,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose source contribution via Delta-E.")
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
