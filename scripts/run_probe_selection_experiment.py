"""Run probe-based source selection experiments on real ETT datasets.

This script implements the user's proposed pseudo-label pipeline directly:

1. Fit a baseline linear probe for each target A using only A's history.
2. Fit source-augmented probes for each candidate source S.
3. Define y(A, S) = Delta_probe(A, S) from validation-loss improvements.
4. Select top-k sources using the probe gains and train a final linear forecaster.

The current implementation treats each individual source variable as a singleton
source set. This provides an ETTh1-ready oracle selection baseline before
learning an energy model that approximates the same ordering.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network.config import DataConfig
from transfer_energy_network.data.datasets import get_dataset_profile
from transfer_energy_network.data.ltsf import load_ltsf_series


@dataclass
class CandidateSeries:
    label: str
    dataset_name: str
    feature_name: str
    values: torch.Tensor
    feature_index: int


@dataclass
class TargetExperimentResult:
    target_label: str
    baseline_val_loss: float
    baseline_test_loss: float
    topk_test_loss: float
    oracle_best_single_test_loss: float
    top_sources: list[str]
    top_source_gains: list[float]


def split_boundaries(length: int, config: DataConfig) -> dict[str, tuple[int, int]]:
    train_end = int(length * config.train_ratio)
    val_end = train_end + int(length * config.val_ratio)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }


def normalize_with_train_stats(values: torch.Tensor, train_end: int) -> torch.Tensor:
    train_values = values[:train_end]
    mean = train_values.mean(dim=0, keepdim=True)
    std = train_values.std(dim=0, keepdim=True).clamp_min(1e-5)
    return (values - mean) / std


def map_cross_dataset_start(
    target_start: int,
    target_split_start: int,
    target_last_start: int,
    source_split_start: int,
    source_last_start: int,
) -> int:
    if source_last_start <= source_split_start:
        return source_split_start
    if target_last_start <= target_split_start:
        return source_split_start
    progress = (target_start - target_split_start) / max(1, target_last_start - target_split_start)
    mapped = source_split_start + progress * (source_last_start - source_split_start)
    return int(round(max(source_split_start, min(source_last_start, mapped))))


def available_start_indices(split: str, values: torch.Tensor, config: DataConfig) -> tuple[list[int], tuple[int, int]]:
    split_start, split_end = split_boundaries(len(values), config)[split]
    max_start = split_end - config.seq_len - config.pred_len + 1
    if max_start <= split_start:
        raise ValueError(f"Split {split} is too short for seq_len={config.seq_len} and pred_len={config.pred_len}")
    starts = list(range(split_start, max_start, config.window_stride))
    if config.max_windows_per_split > 0:
        starts = starts[: config.max_windows_per_split]
    return starts, (split_start, split_end)


def build_candidate_pool(config: DataConfig) -> tuple[torch.Tensor, list[str], list[CandidateSeries]]:
    target_loaded = load_ltsf_series(config)
    target_bounds = split_boundaries(len(target_loaded.values), config)
    target_values = normalize_with_train_stats(target_loaded.values, target_bounds["train"][1])

    candidates: list[CandidateSeries] = []
    for feature_index, feature_name in enumerate(target_loaded.columns):
        candidates.append(
            CandidateSeries(
                label=f"{config.dataset_name}:{feature_name}",
                dataset_name=config.dataset_name,
                feature_name=feature_name,
                values=target_values,
                feature_index=feature_index,
            )
        )

    for source_dataset_name in config.source_pool_datasets:
        loaded = load_ltsf_series(config, dataset_name=source_dataset_name)
        bounds = split_boundaries(len(loaded.values), config)
        normalized_values = normalize_with_train_stats(loaded.values, bounds["train"][1])
        for feature_index, feature_name in enumerate(loaded.columns):
            candidates.append(
                CandidateSeries(
                    label=f"{source_dataset_name}:{feature_name}",
                    dataset_name=source_dataset_name,
                    feature_name=feature_name,
                    values=normalized_values,
                    feature_index=feature_index,
                )
            )
    return target_values, target_loaded.columns, candidates


def build_target_windows(
    values: torch.Tensor,
    target_index: int,
    split: str,
    config: DataConfig,
) -> tuple[torch.Tensor, torch.Tensor, list[int], tuple[int, int]]:
    starts, split_bounds = available_start_indices(split, values, config)
    histories: list[torch.Tensor] = []
    futures: list[torch.Tensor] = []
    for start in starts:
        context_end = start + config.seq_len
        forecast_end = context_end + config.pred_len
        histories.append(values[start:context_end, target_index])
        futures.append(values[context_end:forecast_end, target_index])
    return torch.stack(histories), torch.stack(futures), starts, split_bounds


def build_source_windows(
    candidate: CandidateSeries,
    starts: list[int],
    target_split_bounds: tuple[int, int],
    split: str,
    config: DataConfig,
) -> torch.Tensor:
    split_start, split_end = split_boundaries(len(candidate.values), config)[split]
    source_last_start = split_end - config.seq_len
    target_split_start, target_split_end = target_split_bounds
    target_last_start = target_split_end - config.seq_len - config.pred_len
    histories: list[torch.Tensor] = []
    for target_start in starts:
        if candidate.dataset_name == config.dataset_name:
            source_start = target_start
        else:
            source_start = map_cross_dataset_start(
                target_start=target_start,
                target_split_start=target_split_start,
                target_last_start=target_last_start,
                source_split_start=split_start,
                source_last_start=source_last_start,
            )
        source_end = source_start + config.seq_len
        histories.append(candidate.values[source_start:source_end, candidate.feature_index])
    return torch.stack(histories)


def augment_bias(features: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(features.shape[0], 1, dtype=features.dtype)
    return torch.cat([features, ones], dim=-1)


def fit_ridge_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    features = augment_bias(features)
    gram = features.T @ features
    eye = torch.eye(gram.shape[0], dtype=features.dtype)
    eye[-1, -1] = 0.0
    rhs = features.T @ targets
    return torch.linalg.solve(gram + ridge * eye, rhs)


def predict_probe(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return augment_bias(features) @ weights


def mse_loss(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    return float((targets - predictions).pow(2).mean().item())


def fit_and_eval_probe(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    eval_x: torch.Tensor,
    eval_y: torch.Tensor,
    ridge: float,
) -> tuple[torch.Tensor, float]:
    weights = fit_ridge_probe(train_x, train_y, ridge=ridge)
    predictions = predict_probe(eval_x, weights)
    return weights, mse_loss(eval_y, predictions)


def run_target_experiment(
    target_values: torch.Tensor,
    target_columns: list[str],
    candidates: list[CandidateSeries],
    target_index: int,
    config: DataConfig,
    ridge: float,
    top_k: int,
    positive_only: bool,
) -> TargetExperimentResult:
    train_target_x, train_y, train_starts, train_bounds = build_target_windows(target_values, target_index, "train", config)
    val_target_x, val_y, val_starts, val_bounds = build_target_windows(target_values, target_index, "val", config)
    test_target_x, test_y, test_starts, test_bounds = build_target_windows(target_values, target_index, "test", config)

    _, baseline_val_loss = fit_and_eval_probe(train_target_x, train_y, val_target_x, val_y, ridge=ridge)

    trainval_target_x = torch.cat([train_target_x, val_target_x], dim=0)
    trainval_y = torch.cat([train_y, val_y], dim=0)
    baseline_weights = fit_ridge_probe(trainval_target_x, trainval_y, ridge=ridge)
    baseline_test_loss = mse_loss(test_y, predict_probe(test_target_x, baseline_weights))

    scored_candidates: list[tuple[float, str, CandidateSeries]] = []
    best_single_test_loss = float("inf")
    target_label = target_columns[target_index]

    for candidate in candidates:
        if candidate.dataset_name == config.dataset_name and candidate.feature_index == target_index:
            continue

        train_source_x = build_source_windows(candidate, train_starts, train_bounds, "train", config)
        val_source_x = build_source_windows(candidate, val_starts, val_bounds, "val", config)
        train_features = torch.cat([train_target_x, train_source_x], dim=-1)
        val_features = torch.cat([val_target_x, val_source_x], dim=-1)
        _, source_val_loss = fit_and_eval_probe(train_features, train_y, val_features, val_y, ridge=ridge)
        delta = baseline_val_loss - source_val_loss
        scored_candidates.append((delta, candidate.label, candidate))

        trainval_source_x = torch.cat(
            [
                build_source_windows(candidate, train_starts, train_bounds, "train", config),
                build_source_windows(candidate, val_starts, val_bounds, "val", config),
            ],
            dim=0,
        )
        test_source_x = build_source_windows(candidate, test_starts, test_bounds, "test", config)
        single_weights = fit_ridge_probe(
            torch.cat([trainval_target_x, trainval_source_x], dim=-1),
            trainval_y,
            ridge=ridge,
        )
        single_test_loss = mse_loss(
            test_y,
            predict_probe(torch.cat([test_target_x, test_source_x], dim=-1), single_weights),
        )
        best_single_test_loss = min(best_single_test_loss, single_test_loss)

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    if positive_only:
        scored_candidates = [item for item in scored_candidates if item[0] > 0.0]
    chosen = scored_candidates[:top_k]

    trainval_feature_parts = [trainval_target_x]
    test_feature_parts = [test_target_x]
    for _, _, candidate in chosen:
        trainval_feature_parts.append(
            torch.cat(
                [
                    build_source_windows(candidate, train_starts, train_bounds, "train", config),
                    build_source_windows(candidate, val_starts, val_bounds, "val", config),
                ],
                dim=0,
            )
        )
        test_feature_parts.append(build_source_windows(candidate, test_starts, test_bounds, "test", config))

    if chosen:
        topk_weights = fit_ridge_probe(torch.cat(trainval_feature_parts, dim=-1), trainval_y, ridge=ridge)
        topk_test_loss = mse_loss(test_y, predict_probe(torch.cat(test_feature_parts, dim=-1), topk_weights))
    else:
        topk_test_loss = baseline_test_loss

    return TargetExperimentResult(
        target_label=target_label,
        baseline_val_loss=baseline_val_loss,
        baseline_test_loss=baseline_test_loss,
        topk_test_loss=topk_test_loss,
        oracle_best_single_test_loss=best_single_test_loss,
        top_sources=[label for _, label, _ in chosen],
        top_source_gains=[float(delta) for delta, _, _ in chosen],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run probe-based source selection on ETTh1-style datasets.")
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
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--allow-negative-gains", action="store_true")
    parser.add_argument("--output", default="results/etth1_probe_selection_topk4.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = get_dataset_profile(args.dataset_name)
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
    )

    target_values, target_columns, candidates = build_candidate_pool(config)
    results: list[TargetExperimentResult] = []
    for target_index in range(len(target_columns)):
        results.append(
            run_target_experiment(
                target_values=target_values,
                target_columns=target_columns,
                candidates=candidates,
                target_index=target_index,
                config=config,
                ridge=args.ridge,
                top_k=args.top_k,
                positive_only=not args.allow_negative_gains,
            )
        )

    summary = {
        "config": {
            "dataset_name": args.dataset_name,
            "file_name": args.file_name,
            "source_pool_datasets": args.source_pool_datasets,
            "seq_len": args.seq_len,
            "pred_len": args.pred_len,
            "window_stride": args.window_stride,
            "max_windows_per_split": args.max_windows_per_split,
            "top_k": args.top_k,
            "ridge": args.ridge,
        },
        "aggregate": {
            "mean_baseline_test_loss": sum(item.baseline_test_loss for item in results) / len(results),
            "mean_topk_test_loss": sum(item.topk_test_loss for item in results) / len(results),
            "mean_oracle_best_single_test_loss": sum(item.oracle_best_single_test_loss for item in results) / len(results),
            "mean_topk_gain_over_baseline": sum(item.baseline_test_loss - item.topk_test_loss for item in results) / len(results),
        },
        "per_target": [asdict(item) for item in results],
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"saved_results={output_path}")


if __name__ == "__main__":
    main()
