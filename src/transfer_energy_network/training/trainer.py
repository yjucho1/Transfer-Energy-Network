"""Training utilities for Transfer Energy Network."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import gaussian_nll, transferability_loss


@dataclass
class TrainingConfig:
    """Configuration for a joint forecasting and transferability step."""

    forecast_loss_weight: float = 1.0
    transfer_loss_weight: float = 1.0
    energy_temperature: float = 1.0
    target_temperature: float = 1.0


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Run one joint optimization step."""
    model.train()
    optimizer.zero_grad()

    outputs = model(
        batch["target_context"],
        batch["source_candidates"],
        temperature=config.energy_temperature,
    )

    forecast_loss = gaussian_nll(
        target=batch["forecast_target"],
        mean=outputs["mean"],
        scale=outputs["scale"],
    )
    transfer_loss, target_weights = transferability_loss(
        energies=outputs["energies"],
        validation_delta=batch["validation_delta"],
        temperature=config.target_temperature,
    )

    loss = (
        config.forecast_loss_weight * forecast_loss
        + config.transfer_loss_weight * transfer_loss
    )
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.detach(),
        "forecast_loss": forecast_loss.detach(),
        "transfer_loss": transfer_loss.detach(),
        "weights": outputs["weights"].detach(),
        "target_weights": target_weights.detach(),
        "energies": outputs["energies"].detach(),
        "mean": outputs["mean"].detach(),
        "scale": outputs["scale"].detach(),
    }
