"""Synthetic episodic data for transferability experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..config import DataConfig
from .datasets import get_dataset_profile


@dataclass
class EpisodeBatch:
    target_context: torch.Tensor
    source_candidates: torch.Tensor
    forecast_target: torch.Tensor
    validation_delta: torch.Tensor
    oracle_index: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "target_context": self.target_context,
            "source_candidates": self.source_candidates,
            "forecast_target": self.forecast_target,
            "validation_delta": self.validation_delta,
            "oracle_index": self.oracle_index,
        }


def make_synthetic_batch(config: DataConfig) -> EpisodeBatch:
    """Create one transfer episode batch with a single most-helpful source."""
    _ = get_dataset_profile(config.dataset_name)
    batch_size = config.batch_size
    num_sources = config.num_sources
    target_dim = config.target_dim
    source_dim = config.source_dim
    horizon = config.horizon

    target_context = torch.randn(batch_size, target_dim)
    source_candidates = torch.randn(batch_size, num_sources, source_dim)
    oracle_index = torch.randint(0, num_sources, (batch_size,))
    validation_delta = torch.full(
        (batch_size, num_sources),
        config.harmful_delta,
    )

    shared_dim = min(target_dim, source_dim)
    for row in range(batch_size):
        chosen = oracle_index[row]
        source_candidates[row, chosen, :shared_dim] += target_context[row, :shared_dim]
        validation_delta[row, chosen] = config.helpful_delta

    forecast_signal = source_candidates[torch.arange(batch_size), oracle_index, :horizon]
    forecast_target = (
        config.target_signal_scale * target_context[:, :horizon]
        + config.source_signal_scale * forecast_signal
        + config.noise_scale * torch.randn(batch_size, horizon)
    )

    return EpisodeBatch(
        target_context=target_context,
        source_candidates=source_candidates,
        forecast_target=forecast_target,
        validation_delta=validation_delta,
        oracle_index=oracle_index,
    )


def generate_episode_split(num_episodes: int, config: DataConfig) -> list[EpisodeBatch]:
    return [make_synthetic_batch(config) for _ in range(num_episodes)]
