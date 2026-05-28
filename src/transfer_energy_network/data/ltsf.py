"""Real CSV loaders for long-term forecasting benchmarks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import torch

from ..config import DataConfig
from .datasets import get_dataset_profile
from .episodes import EpisodeBatch
from ..models import PatchTSTAdaptationModel


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


def _fit_ridge_regression(
    features: torch.Tensor,
    targets: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    augmented = torch.cat(
        [features, torch.ones(features.shape[0], 1, dtype=features.dtype)],
        dim=1,
    )
    gram = augmented.T @ augmented
    gram = gram + ridge * torch.eye(gram.shape[0], dtype=features.dtype)
    rhs = augmented.T @ targets
    return torch.linalg.solve(gram, rhs)


def _predict_ridge(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    augmented = torch.cat(
        [features, torch.ones(features.shape[0], 1, dtype=features.dtype)],
        dim=1,
    )
    return augmented @ weights


def _build_adaptation_design(
    target_history: torch.Tensor,
    source_history: torch.Tensor | None,
    lag: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for end in range(lag, target_history.shape[0]):
        target_lags = target_history[end - lag : end]
        if source_history is None:
            feature_vec = target_lags
        else:
            source_lags = source_history[end - lag : end]
            feature_vec = torch.cat([target_lags, source_lags], dim=0)
        features.append(feature_vec)
        targets.append(target_history[end])

    return torch.stack(features, dim=0), torch.stack(targets, dim=0)


def _adaptation_validation_mse(
    target_history: torch.Tensor,
    source_history: torch.Tensor | None,
    config: DataConfig,
) -> float:
    lag = min(config.adaptation_lag, max(1, target_history.shape[0] // 3))
    features, targets = _build_adaptation_design(
        target_history=target_history,
        source_history=source_history,
        lag=lag,
    )

    if features.shape[0] < config.adaptation_min_samples:
        prediction = _repeat_last_forecast(target_history, targets.shape[0])
        return float(torch.mean((targets - prediction).pow(2)).item())

    val_size = max(1, int(features.shape[0] * config.adaptation_val_ratio))
    train_size = features.shape[0] - val_size
    if train_size < 1:
        train_size = features.shape[0] - 1
        val_size = 1

    train_x = features[:train_size]
    train_y = targets[:train_size]
    val_x = features[train_size:]
    val_y = targets[train_size:]

    weights = _fit_ridge_regression(
        features=train_x,
        targets=train_y,
        ridge=config.adaptation_ridge,
    )
    predictions = _predict_ridge(val_x, weights)
    return float(torch.mean((val_y - predictions).pow(2)).item())


def _build_patchtst_examples(
    target_history: torch.Tensor,
    source_history: torch.Tensor | None,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    for end in range(context_length, target_history.shape[0]):
        target_window = target_history[end - context_length : end].unsqueeze(-1)
        if source_history is None:
            feature_window = target_window
        else:
            source_window = source_history[end - context_length : end].unsqueeze(-1)
            feature_window = torch.cat([target_window, source_window], dim=-1)
        features.append(feature_window)
        targets.append(target_history[end])

    return torch.stack(features, dim=0), torch.stack(targets, dim=0)


def _train_patchtst_regressor(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    input_channels: int,
    config: DataConfig,
) -> PatchTSTAdaptationModel:
    model = PatchTSTAdaptationModel(
        input_channels=input_channels,
        context_length=train_x.shape[1],
        patch_len=config.adaptation_patch_len,
        stride=config.adaptation_patch_stride,
        d_model=config.adaptation_d_model,
        n_heads=config.adaptation_n_heads,
        n_layers=config.adaptation_n_layers,
        d_ff=config.adaptation_d_ff,
        dropout=config.adaptation_dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.adaptation_lr)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for _ in range(config.adaptation_epochs):
        optimizer.zero_grad()
        predictions = model(train_x)
        loss = loss_fn(predictions, train_y)
        loss.backward()
        optimizer.step()
    return model


def _patchtst_validation_mse_per_row(
    target_histories: list[torch.Tensor],
    source_histories: list[torch.Tensor] | None,
    config: DataConfig,
) -> torch.Tensor:
    context_length = min(config.adaptation_lag, max(4, target_histories[0].shape[0] // 2))

    train_features: list[torch.Tensor] = []
    train_targets: list[torch.Tensor] = []
    val_features: list[torch.Tensor] = []
    val_targets: list[torch.Tensor] = []
    row_slices: list[tuple[int, int]] = []
    input_channels = 1 if source_histories is None else 2

    running = 0
    for row_index, target_history in enumerate(target_histories):
        source_history = None if source_histories is None else source_histories[row_index]
        features, targets = _build_patchtst_examples(
            target_history=target_history,
            source_history=source_history,
            context_length=context_length,
        )
        if features.shape[0] < config.adaptation_min_samples:
            fallback = torch.full((1,), float(torch.mean((targets - targets.mean()).pow(2)).item()))
            row_slices.append((running, running + 1))
            val_features.append(torch.zeros(1, context_length, input_channels))
            val_targets.append(fallback)
            running += 1
            continue

        val_size = max(1, int(features.shape[0] * config.adaptation_val_ratio))
        train_size = features.shape[0] - val_size
        if train_size < 1:
            train_size = features.shape[0] - 1
            val_size = 1

        train_features.append(features[:train_size])
        train_targets.append(targets[:train_size])
        val_features.append(features[train_size:])
        val_targets.append(targets[train_size:])
        row_slices.append((running, running + val_targets[-1].shape[0]))
        running += val_targets[-1].shape[0]

    train_x = torch.cat(train_features, dim=0)
    train_y = torch.cat(train_targets, dim=0)
    val_x = torch.cat(val_features, dim=0)
    val_y = torch.cat(val_targets, dim=0)

    model = _train_patchtst_regressor(
        train_x=train_x,
        train_y=train_y,
        input_channels=input_channels,
        config=config,
    )
    model.eval()
    with torch.no_grad():
        predictions = model(val_x)
        squared_error = (val_y - predictions).pow(2)

    per_row_mse: list[torch.Tensor] = []
    for start, end in row_slices:
        per_row_mse.append(squared_error[start:end].mean())
    return torch.stack(per_row_mse)


def _compute_batch_validation_deltas(
    target_histories: list[torch.Tensor],
    source_histories_by_row: list[list[torch.Tensor]],
    config: DataConfig,
) -> torch.Tensor:
    if config.adaptation_backbone == "patchtst":
        baseline_mse = _patchtst_validation_mse_per_row(
            target_histories=target_histories,
            source_histories=None,
            config=config,
        )
        num_sources = len(source_histories_by_row[0])
        per_source_deltas: list[torch.Tensor] = []

        for source_slot in range(num_sources):
            source_histories = [row_sources[source_slot] for row_sources in source_histories_by_row]
            source_mse = _patchtst_validation_mse_per_row(
                target_histories=target_histories,
                source_histories=source_histories,
                config=config,
            )
            history_corr = torch.tensor(
                [
                    abs(_safe_corr(source_histories[row], target_histories[row]))
                    for row in range(len(target_histories))
                ],
                dtype=baseline_mse.dtype,
            )
            per_source_deltas.append(baseline_mse - source_mse + 0.02 * history_corr)
        return torch.stack(per_source_deltas, dim=1)

    delta_rows: list[torch.Tensor] = []
    for row, target_history in enumerate(target_histories):
        row_deltas = []
        for source_history in source_histories_by_row[row]:
            row_deltas.append(
                _source_validation_delta(
                    target_history=target_history,
                    future_target=torch.empty(0),
                    source_history=source_history,
                    config=config,
                )
            )
        delta_rows.append(torch.tensor(row_deltas, dtype=torch.float32))
    return torch.stack(delta_rows, dim=0)


def _source_validation_delta(
    target_history: torch.Tensor,
    future_target: torch.Tensor,
    source_history: torch.Tensor,
    config: DataConfig,
) -> float:
    """Estimate transfer utility from actual local adaptation improvement.

    We fit a small per-episode ridge forecaster on target-only history and on
    target-plus-source history, then compare their held-out validation losses.
    A tiny future-alignment bonus is kept only as a tie-breaker.
    """
    del future_target
    baseline_mse = _adaptation_validation_mse(
        target_history=target_history,
        source_history=None,
        config=config,
    )
    source_mse = _adaptation_validation_mse(
        target_history=target_history,
        source_history=source_history,
        config=config,
    )
    improvement = baseline_mse - source_mse

    history_corr = abs(_safe_corr(source_history, target_history))
    return float(improvement + 0.02 * history_corr)


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
    target_histories: list[torch.Tensor] = []
    row_source_histories: list[list[torch.Tensor]] = []

    feature_indices = list(range(normalized_values.shape[1]))
    source_feature_indices = [index for index in feature_indices if index != target_index]

    for start in start_indices:
        context_end = start + config.seq_len
        forecast_end = context_end + config.pred_len

        context = normalized_values[start:context_end]
        future_target = normalized_values[context_end:forecast_end, target_index]
        target_context = context.flatten()
        target_history = context[:, target_index]
        target_histories.append(target_history)

        scored_sources: list[tuple[float, int]] = []
        for source_index in source_feature_indices:
            source_history = context[:, source_index]
            similarity = abs(_safe_corr(source_history, target_history))
            scored_sources.append((similarity, source_index))

        scored_sources.sort(key=lambda item: item[0], reverse=True)
        chosen_sources = [source_index for _, source_index in scored_sources[: config.num_sources]]

        source_candidates: list[torch.Tensor] = []
        current_row_sources: list[torch.Tensor] = []

        for source_index in chosen_sources:
            source_history = context[:, source_index]
            source_candidates.append(source_history)
            current_row_sources.append(source_history)

        source_candidate_tensor = torch.stack(source_candidates, dim=0)
        target_contexts.append(target_context)
        source_candidates_list.append(source_candidate_tensor)
        forecast_targets.append(future_target)
        row_source_histories.append(current_row_sources)

    validation_delta_tensor = _compute_batch_validation_deltas(
        target_histories=target_histories,
        source_histories_by_row=row_source_histories,
        config=config,
    )
    oracle_indices = validation_delta_tensor.argmax(dim=1)

    return EpisodeBatch(
        target_context=torch.stack(target_contexts, dim=0),
        source_candidates=torch.stack(source_candidates_list, dim=0),
        forecast_target=torch.stack(forecast_targets, dim=0),
        validation_delta=validation_delta_tensor,
        oracle_index=oracle_indices.to(dtype=torch.long),
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
