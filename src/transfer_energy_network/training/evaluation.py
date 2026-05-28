"""Evaluation and experiment-summary helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..data import EpisodeBatch
from ..models.baselines import correlation_weights, oracle_weights, uniform_weights
from .losses import gaussian_crps, gaussian_quantile, pinball_loss


@dataclass
class EpochMetrics:
    forecast_mse: float
    forecast_mae: float
    forecast_crps: float
    forecast_q10_loss: float
    forecast_q50_loss: float
    forecast_q90_loss: float
    forecast_mean_quantile_loss: float
    top1_alignment: float
    oracle_hit_rate: float


@dataclass
class ExperimentSummary:
    config: dict
    train_history: list[dict]
    validation_history: list[dict]
    test_metrics: dict


def weighted_source_pool(
    weights: torch.Tensor,
    source_candidates: torch.Tensor,
    target_dim: int,
) -> torch.Tensor:
    pooled = torch.sum(weights.unsqueeze(-1) * source_candidates, dim=1)
    return pooled[:, :target_dim]


def build_forecast_from_weights(
    batch: EpisodeBatch,
    weights: torch.Tensor,
    config: ExperimentConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    pooled_sources = weighted_source_pool(
        weights=weights,
        source_candidates=batch.source_candidates,
        target_dim=batch.source_candidates.shape[-1],
    )
    pred_len = batch.forecast_target.shape[-1]
    last_value = pooled_sources[:, -1:].expand(-1, pred_len)
    median = last_value
    return median, torch.full_like(median, 0.1)


def compute_point_metrics(
    target: torch.Tensor,
    median_forecast: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    squared_error = (target - median_forecast).pow(2)
    absolute_error = (target - median_forecast).abs()
    return squared_error.mean(), absolute_error.mean()


def evaluate_selector(
    selector: str,
    episodes: list[EpisodeBatch],
    config: ExperimentConfig,
    model: torch.nn.Module | None = None,
) -> EpochMetrics:
    mses: list[float] = []
    maes: list[float] = []
    crps_scores: list[float] = []
    q10_losses: list[float] = []
    q50_losses: list[float] = []
    q90_losses: list[float] = []
    mean_q_losses: list[float] = []
    alignments: list[float] = []
    oracle_hits: list[float] = []

    for batch in episodes:
        if selector == "ten":
            if model is None:
                raise ValueError("TEN evaluation requires a model")
            outputs = model(
                batch.target_context,
                batch.source_candidates,
                temperature=config.model.energy_temperature,
            )
            weights = outputs["weights"].detach()
            top1 = weights.argmax(dim=-1)
            median = outputs["mean"].detach()
            scale = outputs["scale"].detach()
        elif selector == "correlation":
            weights = correlation_weights(
                batch.target_context,
                batch.source_candidates,
                temperature=config.model.energy_temperature,
            )
            top1 = weights.argmax(dim=-1)
            median, scale = build_forecast_from_weights(batch, weights, config)
        elif selector == "uniform":
            weights = uniform_weights(batch.target_context, batch.source_candidates)
            top1 = weights.argmax(dim=-1)
            median, scale = build_forecast_from_weights(batch, weights, config)
        elif selector == "oracle":
            weights = oracle_weights(batch.validation_delta)
            top1 = weights.argmax(dim=-1)
            median, scale = build_forecast_from_weights(batch, weights, config)
        else:
            raise ValueError(f"Unknown selector: {selector}")
        oracle_top1 = batch.validation_delta.argmax(dim=-1)
        mse, mae = compute_point_metrics(batch.forecast_target, median)
        crps = gaussian_crps(batch.forecast_target, median, scale)
        q10 = gaussian_quantile(median, scale, 0.1)
        q50 = gaussian_quantile(median, scale, 0.5)
        q90 = gaussian_quantile(median, scale, 0.9)
        q10_loss = pinball_loss(batch.forecast_target, q10, 0.1)
        q50_loss = pinball_loss(batch.forecast_target, q50, 0.5)
        q90_loss = pinball_loss(batch.forecast_target, q90, 0.9)
        mean_q_loss = (q10_loss + q50_loss + q90_loss) / 3.0

        mses.append(float(mse.item()))
        maes.append(float(mae.item()))
        crps_scores.append(float(crps.item()))
        q10_losses.append(float(q10_loss.item()))
        q50_losses.append(float(q50_loss.item()))
        q90_losses.append(float(q90_loss.item()))
        mean_q_losses.append(float(mean_q_loss.item()))
        alignments.append(float((top1 == oracle_top1).float().mean().item()))
        oracle_hits.append(float((top1 == batch.oracle_index).float().mean().item()))

    return EpochMetrics(
        forecast_mse=sum(mses) / len(mses),
        forecast_mae=sum(maes) / len(maes),
        forecast_crps=sum(crps_scores) / len(crps_scores),
        forecast_q10_loss=sum(q10_losses) / len(q10_losses),
        forecast_q50_loss=sum(q50_losses) / len(q50_losses),
        forecast_q90_loss=sum(q90_losses) / len(q90_losses),
        forecast_mean_quantile_loss=sum(mean_q_losses) / len(mean_q_losses),
        top1_alignment=sum(alignments) / len(alignments),
        oracle_hit_rate=sum(oracle_hits) / len(oracle_hits),
    )


def save_experiment_summary(path: str | Path, summary: ExperimentSummary) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(summary), fh, indent=2)
