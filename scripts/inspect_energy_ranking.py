"""Inspect source-wise energy and final weights for one episode."""

from __future__ import annotations

import argparse
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


def rank_asc(values: list[float]) -> list[int]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0] * len(values)
    for rank, idx in enumerate(order, start=1):
        ranks[idx] = rank
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect energy ranking on one test episode.")
    parser.add_argument("config", help="Path to TEN config")
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    config = load_experiment_config(Path(args.config))
    model, _, _, test_episodes, _ = load_best_ten_model(config)
    batch = test_episodes[args.batch_index]

    with torch.no_grad():
        outputs = model(
            batch.target_context,
            batch.source_candidates,
            temperature=config.model.energy_temperature,
        )

    sample_index = args.sample_index
    labels = batch.source_labels[sample_index] if batch.source_labels is not None else [
        f"source_{i}" for i in range(batch.source_candidates.shape[1])
    ]
    energies = outputs["energies"][sample_index].cpu().tolist()
    weights = outputs["weights"][sample_index].cpu().tolist()

    energy_ranks = rank_asc(energies)
    weight_ranks = rank_asc([-value for value in weights])

    print(f"config={args.config}")
    print(f"batch_index={args.batch_index} sample_index={sample_index}")
    print("")
    print(f"{'idx':>3}  {'source':<20} {'energy':>10} {'weight':>10} {'r_energy':>8} {'r_weight':>8}")
    for idx, label in enumerate(labels):
        print(
            f"{idx:>3}  {label:<20} "
            f"{energies[idx]:>10.6f} {weights[idx]:>10.6f} "
            f"{energy_ranks[idx]:>8} {weight_ranks[idx]:>8}"
        )


if __name__ == "__main__":
    main()
