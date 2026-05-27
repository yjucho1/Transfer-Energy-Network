"""Evaluation and experiment-summary helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import torch

from ..config import ExperimentConfig
from ..data import EpisodeBatch
from ..models.baselines import correlation_weights, oracle_weights, uniform_weights
from .losses import gaussian_nll


@dataclass
class EpochMetrics:
    forecast_nll: float
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
    mean = last_value
    scale = torch.full_like(mean, 0.1)
    return mean, scale


def evaluate_selector(
    selector: str,
    episodes: list[EpisodeBatch],
    config: ExperimentConfig,
    model: torch.nn.Module | None = None,
) -> EpochMetrics:
    nlls: list[float] = []
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
            nll = gaussian_nll(batch.forecast_target, outputs["mean"].detach(), outputs["scale"].detach())
        elif selector == "correlation":
            weights = correlation_weights(
                batch.target_context,
                batch.source_candidates,
                temperature=config.model.energy_temperature,
            )
            top1 = weights.argmax(dim=-1)
            mean, scale = build_forecast_from_weights(batch, weights, config)
            nll = gaussian_nll(batch.forecast_target, mean, scale)
        elif selector == "uniform":
            weights = uniform_weights(batch.target_context, batch.source_candidates)
            top1 = weights.argmax(dim=-1)
            mean, scale = build_forecast_from_weights(batch, weights, config)
            nll = gaussian_nll(batch.forecast_target, mean, scale)
        elif selector == "oracle":
            weights = oracle_weights(batch.validation_delta)
            top1 = weights.argmax(dim=-1)
            mean, scale = build_forecast_from_weights(batch, weights, config)
            nll = gaussian_nll(batch.forecast_target, mean, scale)
        else:
            raise ValueError(f"Unknown selector: {selector}")
        oracle_top1 = batch.validation_delta.argmax(dim=-1)

        nlls.append(float(nll.item()))
        alignments.append(float((top1 == oracle_top1).float().mean().item()))
        oracle_hits.append(float((top1 == batch.oracle_index).float().mean().item()))

    return EpochMetrics(
        forecast_nll=sum(nlls) / len(nlls),
        top1_alignment=sum(alignments) / len(alignments),
        oracle_hit_rate=sum(oracle_hits) / len(oracle_hits),
    )


def save_experiment_summary(path: str | Path, summary: ExperimentSummary) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(asdict(summary), fh, indent=2)
