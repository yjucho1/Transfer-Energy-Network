"""Loss functions for probabilistic forecasting and transferability learning."""

from __future__ import annotations

import math

import torch


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


def gaussian_mean_quantile_loss(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> torch.Tensor:
    """Average pinball loss over analytic Gaussian quantiles."""
    losses = []
    for quantile in quantiles:
        prediction = gaussian_quantile(mean, scale, quantile)
        losses.append(pinball_loss(target, prediction, quantile))
    return torch.stack(losses).mean()


def transferability_loss(
    energies: torch.Tensor,
    validation_delta: torch.Tensor,
    margin: float = 0.2,
    delta_threshold: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rank low-energy sources ahead of less transferable ones with hinge loss."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    if delta_threshold < 0:
        raise ValueError("delta_threshold must be non-negative")

    delta_diff = validation_delta.unsqueeze(-1) - validation_delta.unsqueeze(-2)
    pair_mask = delta_diff > delta_threshold

    energy_diff = energies.unsqueeze(-1) - energies.unsqueeze(-2)
    pair_losses = torch.relu(energy_diff + margin)
    masked_pair_losses = pair_losses * pair_mask.float()

    valid_pair_count = pair_mask.float().sum()
    if valid_pair_count.item() == 0:
        zero = energies.new_zeros(())
        return zero, pair_mask.float()

    loss = masked_pair_losses.sum() / valid_pair_count
    return loss, pair_mask.float()


def trajectory_energy_loss(
    energy_gt: torch.Tensor,
    energy_pred: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """Encourage the ground-truth future to have lower energy than the current forecast."""
    if margin < 0:
        raise ValueError("margin must be non-negative")
    return torch.relu(energy_gt - energy_pred + margin).mean()
