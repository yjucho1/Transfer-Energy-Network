"""Build sample-level pseudo-labels for source ranking experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network.config import DataConfig
from transfer_energy_network.data import generate_ltsf_episode_split
from scripts.run_probe_selection_experiment import (
    build_candidate_pool,
    build_source_windows,
    build_target_windows,
    fit_ridge_probe,
    predict_probe,
)


def per_sample_mse(target: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    return (target - prediction).pow(2).mean(dim=-1)


def build_split_payload(
    split: str,
    episodes: list,
    gain_lookup: dict[str, tuple[dict[str, int], torch.Tensor]],
) -> dict:
    counters = {target_label: 0 for target_label in gain_lookup}
    batches = []
    for batch in episodes:
        if batch.source_labels is None or batch.metadata is None:
            raise ValueError("Real episodes with source_labels and metadata are required.")
        target_labels = batch.metadata["target_labels"]
        pseudo_gain = torch.empty(
            batch.target_context.shape[0],
            batch.source_candidates.shape[1],
            dtype=batch.target_context.dtype,
        )
        for row_index, target_label in enumerate(target_labels):
            label_to_index, gain_matrix = gain_lookup[target_label]
            sample_index = counters[target_label]
            counters[target_label] += 1
            for source_index, source_label in enumerate(batch.source_labels[row_index]):
                pseudo_gain[row_index, source_index] = gain_matrix[sample_index, label_to_index[source_label]]
        batches.append(
            {
                "target_context": batch.target_context,
                "source_candidates": batch.source_candidates,
                "forecast_target": batch.forecast_target,
                "pseudo_gain": pseudo_gain,
                "source_labels": batch.source_labels,
                "metadata": batch.metadata,
            }
        )
    return {
        "split": split,
        "num_batches": len(batches),
        "batches": batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build sample-level probe pseudo-labels.")
    parser.add_argument("--dataset-name", default="etth1")
    parser.add_argument("--file-name", default="ETT-small/ETTh1.csv")
    parser.add_argument("--source-pool-datasets", nargs="+", default=["etth2", "ettm1", "ettm2"])
    parser.add_argument("--dataset-bundle", default="./datasets")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--window-stride", type=int, default=8)
    parser.add_argument("--max-windows-per-split", type=int, default=256)
    parser.add_argument("--num-sources", type=int, default=27)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--output-dir", default="results/pseudolabels/etth1_sample_level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataConfig(
        dataset_bundle=args.dataset_bundle,
        dataset_name=args.dataset_name,
        file_name=args.file_name,
        source_pool_datasets=args.source_pool_datasets,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        window_stride=args.window_stride,
        max_windows_per_split=args.max_windows_per_split,
        num_sources=args.num_sources,
    )

    target_values, target_columns, candidates = build_candidate_pool(config)
    split_gains: dict[str, dict[str, tuple[dict[str, int], torch.Tensor]]] = {
        "train": {},
        "val": {},
        "test": {},
    }

    for target_index, target_label in enumerate(target_columns):
        train_target_x, train_y, train_starts, train_bounds = build_target_windows(target_values, target_index, "train", config)
        val_target_x, val_y, val_starts, val_bounds = build_target_windows(target_values, target_index, "val", config)
        test_target_x, test_y, test_starts, test_bounds = build_target_windows(target_values, target_index, "test", config)

        base_weights = fit_ridge_probe(train_target_x, train_y, ridge=args.ridge)
        base_predictions = {
            "train": predict_probe(train_target_x, base_weights),
            "val": predict_probe(val_target_x, base_weights),
            "test": predict_probe(test_target_x, base_weights),
        }
        base_losses = {
            "train": per_sample_mse(train_y, base_predictions["train"]),
            "val": per_sample_mse(val_y, base_predictions["val"]),
            "test": per_sample_mse(test_y, base_predictions["test"]),
        }

        candidate_labels = []
        train_gain_columns = []
        val_gain_columns = []
        test_gain_columns = []

        for candidate in candidates:
            if candidate.dataset_name == config.dataset_name and candidate.feature_index == target_index:
                continue

            train_source_x = build_source_windows(candidate, train_starts, train_bounds, "train", config)
            val_source_x = build_source_windows(candidate, val_starts, val_bounds, "val", config)
            test_source_x = build_source_windows(candidate, test_starts, test_bounds, "test", config)

            probe_weights = fit_ridge_probe(torch.cat([train_target_x, train_source_x], dim=-1), train_y, ridge=args.ridge)
            train_predictions = predict_probe(torch.cat([train_target_x, train_source_x], dim=-1), probe_weights)
            val_predictions = predict_probe(torch.cat([val_target_x, val_source_x], dim=-1), probe_weights)
            test_predictions = predict_probe(torch.cat([test_target_x, test_source_x], dim=-1), probe_weights)

            train_gain_columns.append(base_losses["train"] - per_sample_mse(train_y, train_predictions))
            val_gain_columns.append(base_losses["val"] - per_sample_mse(val_y, val_predictions))
            test_gain_columns.append(base_losses["test"] - per_sample_mse(test_y, test_predictions))
            candidate_labels.append(candidate.label)

        label_to_index = {label: index for index, label in enumerate(candidate_labels)}
        split_gains["train"][target_label] = (label_to_index, torch.stack(train_gain_columns, dim=1))
        split_gains["val"][target_label] = (label_to_index, torch.stack(val_gain_columns, dim=1))
        split_gains["test"][target_label] = (label_to_index, torch.stack(test_gain_columns, dim=1))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        episodes = generate_ltsf_episode_split(split, config)
        payload = build_split_payload(split, episodes, split_gains[split])
        output_path = output_dir / f"{split}.pt"
        torch.save(payload, output_path)
        print(f"saved_{split}={output_path}")


if __name__ == "__main__":
    main()
