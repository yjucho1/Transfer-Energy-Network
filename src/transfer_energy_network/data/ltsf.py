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


def _safe_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = torch.sqrt((x_centered.pow(2).sum() * y_centered.pow(2).sum()).clamp_min(1e-8))
    return float((x_centered * y_centered).sum().item() / denom.item())

def load_ltsf_series(
    config: DataConfig,
    dataset_name: str | None = None,
    file_name: str | None = None,
) -> LoadedSeries:
    """Load a ProbTS-prepared LTSF CSV file into a tensor."""
    resolved_dataset_name = dataset_name or config.dataset_name
    profile = get_dataset_profile(resolved_dataset_name)
    resolved_file_name = file_name
    if resolved_file_name is None:
        if resolved_dataset_name == config.dataset_name:
            resolved_file_name = config.file_name
        else:
            resolved_file_name = profile.file_name
    csv_path = Path(config.dataset_bundle) / resolved_file_name
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
    return LoadedSeries(
        values=values,
        columns=feature_columns,
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


def _map_cross_dataset_start(
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


def _load_source_pool(
    split: str,
    config: DataConfig,
) -> list[tuple[str, torch.Tensor, int, str, tuple[int, int]]]:
    source_pool: list[tuple[str, torch.Tensor, int, str, tuple[int, int]]] = []
    for source_dataset_name in config.source_pool_datasets:
        loaded = load_ltsf_series(
            config,
            dataset_name=source_dataset_name,
        )
        source_boundaries = _split_boundaries(len(loaded.values), config)
        source_train_end = source_boundaries["train"][1]
        normalized_values = _normalize_with_train_stats(loaded.values, source_train_end)
        for column_index, column_name in enumerate(loaded.columns):
            source_pool.append(
                (
                    source_dataset_name,
                    normalized_values,
                    column_index,
                    column_name,
                    source_boundaries[split],
                )
            )
    return source_pool


def _build_window_batch(
    normalized_values: torch.Tensor,
    columns: list[str],
    config: DataConfig,
    row_specs: list[tuple[int, int]],
    split_start: int,
    split_end: int,
    source_pool: list[tuple[str, torch.Tensor, int, str, tuple[int, int]]] | None = None,
) -> EpisodeBatch:
    target_contexts: list[torch.Tensor] = []
    source_candidates_list: list[torch.Tensor] = []
    forecast_targets: list[torch.Tensor] = []
    row_source_labels: list[list[str]] = []
    row_target_labels: list[str] = []

    feature_indices = list(range(normalized_values.shape[1]))
    target_last_start = split_end - config.seq_len - config.pred_len

    for start, target_index in row_specs:
        context_end = start + config.seq_len
        forecast_end = context_end + config.pred_len

        context = normalized_values[start:context_end]
        future_target = normalized_values[context_end:forecast_end, target_index]
        target_history = context[:, target_index]
        target_context = target_history.clone()
        source_feature_indices = [index for index in feature_indices if index != target_index]

        source_candidates: list[torch.Tensor] = []
        current_row_labels: list[str] = []

        if source_pool:
            scored_pool: list[tuple[float, torch.Tensor, str]] = []

            for source_index in source_feature_indices:
                source_history = context[:, source_index]
                similarity = abs(_safe_corr(source_history, target_history))
                scored_pool.append(
                    (
                        similarity,
                        source_history,
                        f"{config.dataset_name}:{columns[source_index]}",
                    )
                )

            for source_name, source_values, source_feature_index, column_name, (source_split_start, source_split_end) in source_pool:
                source_last_start = source_split_end - config.seq_len
                source_start = _map_cross_dataset_start(
                    target_start=start,
                    target_split_start=split_start,
                    target_last_start=target_last_start,
                    source_split_start=source_split_start,
                    source_last_start=source_last_start,
                )
                source_context_end = source_start + config.seq_len
                source_history = source_values[source_start:source_context_end, source_feature_index]
                similarity = abs(_safe_corr(source_history, target_history))
                scored_pool.append((similarity, source_history, f"{source_name}:{column_name}"))

            scored_pool.sort(key=lambda item: item[0], reverse=True)
            for _, source_history, source_label in scored_pool[: config.num_sources]:
                source_candidates.append(source_history)
                current_row_labels.append(source_label)
        else:
            scored_sources: list[tuple[float, int]] = []
            for source_index in source_feature_indices:
                source_history = context[:, source_index]
                similarity = abs(_safe_corr(source_history, target_history))
                scored_sources.append((similarity, source_index))

            scored_sources.sort(key=lambda item: item[0], reverse=True)
            chosen_sources = [source_index for _, source_index in scored_sources[: config.num_sources]]

            for source_index in chosen_sources:
                source_history = context[:, source_index]
                source_candidates.append(source_history)
                current_row_labels.append(f"{config.dataset_name}:{columns[source_index]}")

        source_candidate_tensor = torch.stack(source_candidates, dim=0)
        target_contexts.append(target_context)
        source_candidates_list.append(source_candidate_tensor)
        forecast_targets.append(future_target)
        row_source_labels.append(current_row_labels)
        row_target_labels.append(columns[target_index])

    return EpisodeBatch(
        target_context=torch.stack(target_contexts, dim=0),
        source_candidates=torch.stack(source_candidates_list, dim=0),
        forecast_target=torch.stack(forecast_targets, dim=0),
        source_labels=row_source_labels,
        metadata={
            "split_start": split_start,
            "split_end": split_end,
            "target_labels": row_target_labels,
        },
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
    row_specs = [
        (start, target_index)
        for start in start_indices
        for target_index in range(len(loaded.columns))
    ]

    source_pool = _load_source_pool(split, config) if config.source_pool_datasets else None

    batches: list[EpisodeBatch] = []
    for offset in range(0, len(row_specs), config.batch_size):
        batch_specs = row_specs[offset : offset + config.batch_size]
        if len(batch_specs) < config.batch_size:
            continue
        batches.append(
            _build_window_batch(
                normalized_values=normalized_values,
                columns=loaded.columns,
                config=config,
                row_specs=batch_specs,
                split_start=split_start,
                split_end=split_end,
                source_pool=source_pool,
            )
        )
    return batches
