"""Real CSV loaders for long-term forecasting benchmarks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import DataConfig
from .datasets import get_dataset_profile
from .episodes import EpisodeBatch


@dataclass
class LoadedSeries:
    values: torch.Tensor
    columns: list[str]
    target_index: int


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.pow(2).sum() * y_centered.pow(2).sum()).clamp_min(1e-8))
    return float((x_centered * y_centered).sum().item() / denom.item())


def _repeat_last_forecast(history: torch.Tensor, pred_len: int) -> torch.Tensor:
    return history[-1:].expand(pred_len)


def _project_source_to_target(
    target_history: torch.Tensor,
    source_history: torch.Tensor,
    pred_len: int,
) -> torch.Tensor:
    """Project source dynamics into the target scale with a simple affine match.

    This provides a stronger proxy than raw correlation by asking whether a
    source-implied forecast can beat a target-only persistence baseline.
    """
    target_mean = target_history.mean()
    source_mean = source_history.mean()
    target_std = target_history.std().clamp_min(1e-5)
    source_std = source_history.std().clamp_min(1e-5)

    normalized_source = (source_history - source_mean) / source_std
    aligned_source = normalized_source * target_std + target_mean

    if source_history.shape[0] == 1:
        slope = torch.tensor(0.0, dtype=source_history.dtype)
    else:
        slope = (aligned_source[-1] - aligned_source[0]) / (source_history.shape[0] - 1)

    steps = torch.arange(1, pred_len + 1, dtype=source_history.dtype)
    return aligned_source[-1] + slope * steps


def _source_validation_delta(
    target_history: torch.Tensor,
    future_target: torch.Tensor,
    source_history: torch.Tensor,
) -> float:
    """Estimate transfer utility by comparing forecast quality to a naive target baseline."""
    baseline_forecast = _repeat_last_forecast(target_history, future_target.shape[0])
    source_forecast = _project_source_to_target(
        target_history=target_history,
        source_history=source_history,
        pred_len=future_target.shape[0],
    )

    baseline_mse = torch.mean((future_target - baseline_forecast).pow(2))
    source_mse = torch.mean((future_target - source_forecast).pow(2))
    improvement = baseline_mse - source_mse

    history_corr = abs(_safe_corr(source_history, target_history))
    trend_horizon = min(target_history.shape[0], future_target.shape[0])
    trend_corr = abs(_safe_corr(source_history[-trend_horizon:], future_target[:trend_horizon]))

    # Improvement is the primary signal; correlations stabilize ranking when
    # multiple candidates have similar proxy forecast error.
    return float(improvement.item() + 0.1 * history_corr + 0.05 * trend_corr)


def load_ltsf_series(config: DataConfig) -> LoadedSeries:
    """Load a ProbTS-prepared LTSF CSV file into a tensor."""
    profile = get_dataset_profile(config.dataset_name)
    csv_path = Path(config.dataset_bundle) / profile.file_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        feature_columns = header[1:]
        rows: list[list[float]] = []
        for row in reader:
            if not row:
                continue
            rows.append([float(value) for value in row[1:]])

    values = torch.tensor(rows, dtype=torch.float32)
    if config.target in feature_columns:
        target_index = feature_columns.index(config.target)
    else:
        target_index = len(feature_columns) - 1

    return LoadedSeries(
        values=values,
        columns=feature_columns,
        target_index=target_index,
    )


def _split_boundaries(length: int, config: DataConfig) -> dict[str, tuple[int, int]]:
    train_end = int(length * config.train_ratio)
    val_end = train_end + int(length * config.val_ratio)
    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, length),
    }


def _normalize_with_train_stats(values: torch.Tensor, train_end: int) -> torch.Tensor:
    train_values = values[:train_end]
    mean = train_values.mean(dim=0, keepdim=True)
    std = train_values.std(dim=0, keepdim=True).clamp_min(1e-5)
    return (values - mean) / std


def _build_window_batch(
    normalized_values: torch.Tensor,
    target_index: int,
    config: DataConfig,
    start_indices: list[int],
) -> EpisodeBatch:
    target_contexts: list[torch.Tensor] = []
    source_candidates_list: list[torch.Tensor] = []
    forecast_targets: list[torch.Tensor] = []
    validation_deltas: list[torch.Tensor] = []
    oracle_indices: list[int] = []

    feature_indices = list(range(normalized_values.shape[1]))
    source_feature_indices = [index for index in feature_indices if index != target_index]

    for start in start_indices:
        context_end = start + config.seq_len
        forecast_end = context_end + config.pred_len

        context = normalized_values[start:context_end]
        future_target = normalized_values[context_end:forecast_end, target_index]
        target_context = context.flatten()
        target_history = context[:, target_index]

        scored_sources: list[tuple[float, int]] = []
        for source_index in source_feature_indices:
            source_history = context[:, source_index]
            similarity = abs(_safe_corr(source_history, target_history))
            scored_sources.append((similarity, source_index))

        scored_sources.sort(key=lambda item: item[0], reverse=True)
        chosen_sources = [source_index for _, source_index in scored_sources[: config.num_sources]]

        source_candidates: list[torch.Tensor] = []
        source_deltas: list[float] = []

        for source_index in chosen_sources:
            source_history = context[:, source_index]
            source_candidates.append(source_history)
            delta = _source_validation_delta(
                target_history=target_history,
                future_target=future_target,
                source_history=source_history,
            )
            source_deltas.append(delta)

        source_candidate_tensor = torch.stack(source_candidates, dim=0)
        validation_delta = torch.tensor(source_deltas, dtype=torch.float32)
        oracle_index = int(validation_delta.argmax().item())

        target_contexts.append(target_context)
        source_candidates_list.append(source_candidate_tensor)
        forecast_targets.append(future_target)
        validation_deltas.append(validation_delta)
        oracle_indices.append(oracle_index)

    return EpisodeBatch(
        target_context=torch.stack(target_contexts, dim=0),
        source_candidates=torch.stack(source_candidates_list, dim=0),
        forecast_target=torch.stack(forecast_targets, dim=0),
        validation_delta=torch.stack(validation_deltas, dim=0),
        oracle_index=torch.tensor(oracle_indices, dtype=torch.long),
    )


def generate_ltsf_episode_split(split: str, config: DataConfig) -> list[EpisodeBatch]:
    """Generate batched episodes directly from a real LTSF CSV file."""
    loaded = load_ltsf_series(config)
    boundaries = _split_boundaries(len(loaded.values), config)
    train_end = boundaries["train"][1]
    normalized_values = _normalize_with_train_stats(loaded.values, train_end)

    split_start, split_end = boundaries[split]
    max_start = split_end - config.seq_len - config.pred_len + 1
    if max_start <= split_start:
        raise ValueError(f"Split {split} is too short for seq_len={config.seq_len} and pred_len={config.pred_len}")

    start_indices = list(range(split_start, max_start, config.window_stride))
    if config.max_windows_per_split > 0:
        start_indices = start_indices[: config.max_windows_per_split]

    batches: list[EpisodeBatch] = []
    for offset in range(0, len(start_indices), config.batch_size):
        batch_indices = start_indices[offset : offset + config.batch_size]
        if len(batch_indices) < config.batch_size:
            continue
        batches.append(
            _build_window_batch(
                normalized_values=normalized_values,
                target_index=loaded.target_index,
                config=config,
                start_indices=batch_indices,
            )
        )
    return batches
