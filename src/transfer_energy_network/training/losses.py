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


def gaussian_crps(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Average CRPS for a Gaussian predictive distribution."""
    scale = scale.clamp_min(1e-6)
    z = (target - mean) / scale
    pdf = torch.exp(-0.5 * z.pow(2)) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
    crps = scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / math.sqrt(math.pi))
    return crps.mean()


def gaussian_quantile(
    mean: torch.Tensor,
    scale: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Closed-form Gaussian quantile forecast."""
    z_table = {
        0.1: -1.2815515655446004,
        0.5: 0.0,
        0.9: 1.2815515655446004,
    }
    if quantile not in z_table:
        raise ValueError(f"Unsupported quantile {quantile}; expected one of {sorted(z_table)}")
    return mean + scale * z_table[quantile]


def pinball_loss(
    target: torch.Tensor,
    prediction: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Average pinball loss for one quantile."""
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


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
