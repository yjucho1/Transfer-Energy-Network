"""Loss functions for probabilistic forecasting and transferability learning."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gaussian_nll(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Average Gaussian negative log-likelihood."""
    variance = scale.pow(2)
    log_term = torch.log(variance)
    squared_error = (target - mean).pow(2) / variance
    return 0.5 * (log_term + squared_error + math.log(2.0 * math.pi)).mean()


def transferability_targets(
    validation_delta: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Convert validation improvements into soft supervision targets."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(validation_delta / temperature, dim=-1)


def transferability_loss(
    energies: torch.Tensor,
    validation_delta: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match low-energy assignments to sources with larger validation gains."""
    target_weights = transferability_targets(
        validation_delta=validation_delta,
        temperature=temperature,
    )
    log_probs = F.log_softmax(-energies / temperature, dim=-1)
    loss = -(target_weights * log_probs).sum(dim=-1).mean()
    return loss, target_weights
