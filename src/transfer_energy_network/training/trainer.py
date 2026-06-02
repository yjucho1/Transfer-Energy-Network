"""Training utilities for Transfer Energy Network."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import gaussian_crps, gaussian_mean_quantile_loss, gaussian_nll, trajectory_energy_loss


@dataclass
class TrainingConfig:
    """Configuration for a joint forecasting and transferability step."""

    forecast_loss_weight: float = 1.0
    forecast_loss_type: str = "nll"
    energy_temperature: float = 1.0
    target_temperature: float = 1.0
    energy_margin: float = 0.2
    trajectory_energy_weight: float = 0.1


def compute_forecast_loss(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "nll":
        return gaussian_nll(target=target, mean=mean, scale=scale)
    if loss_type == "crps":
        return gaussian_crps(target=target, mean=mean, scale=scale)
    if loss_type == "quantile":
        return gaussian_mean_quantile_loss(target=target, mean=mean, scale=scale)
    raise ValueError(f"Unsupported forecast_loss_type: {loss_type}")


def _set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


def train_step(
    model: torch.nn.Module,
    optimizer_task: torch.optim.Optimizer,
    optimizer_loss: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: TrainingConfig,
) -> dict[str, torch.Tensor]:
    """Run one SEAL-style alternating optimization step."""
    model.train()
    loss_modules = [model.future_encoder, model.trajectory_energy_mlp]
    task_modules = [model.target_encoder, model.source_encoder, model.energy_mlp, model.forecast_head]

    # 1) Update loss-net / energy heads while task-net is frozen.
    for module in task_modules:
        _set_requires_grad(module, False)
    for module in loss_modules:
        _set_requires_grad(module, True)

    optimizer_loss.zero_grad()
    loss_outputs = model(
        batch["target_context"],
        batch["source_candidates"],
        temperature=config.energy_temperature,
        forecast_target=batch["forecast_target"],
    )
    energy_loss = trajectory_energy_loss(
        energy_gt=loss_outputs["trajectory_energy_gt"],
        energy_pred=loss_outputs["trajectory_energy_pred"],
        margin=config.energy_margin,
    )
    loss_term = energy_loss
    loss_term.backward()
    optimizer_loss.step()

    # 2) Update task-net while loss-net is frozen.
    for module in task_modules:
        _set_requires_grad(module, True)
    for module in loss_modules:
        _set_requires_grad(module, False)

    optimizer_task.zero_grad()
    task_outputs = model(
        batch["target_context"],
        batch["source_candidates"],
        temperature=config.energy_temperature,
        forecast_target=batch["forecast_target"],
    )
    forecast_loss = compute_forecast_loss(
        target=batch["forecast_target"],
        mean=task_outputs["mean"],
        scale=task_outputs["scale"],
        loss_type=config.forecast_loss_type,
    )
    trajectory_energy_term = config.trajectory_energy_weight * task_outputs["trajectory_energy_pred"].mean()
    forecast_term = config.forecast_loss_weight * forecast_loss
    task_total = forecast_term + trajectory_energy_term
    task_total.backward()
    optimizer_task.step()

    for module in loss_modules:
        _set_requires_grad(module, True)

    return {
        "loss": (task_total + loss_term).detach(),
        "forecast_loss": forecast_loss.detach(),
        "energy_loss": energy_loss.detach(),
        "trajectory_energy_loss": trajectory_energy_term.detach(),
        "weights": task_outputs["weights"].detach(),
        "energies": task_outputs["energies"].detach(),
        "trajectory_energy_pred": task_outputs["trajectory_energy_pred"].detach(),
        "trajectory_energy_gt": task_outputs["trajectory_energy_gt"].detach(),
        "mean": task_outputs["mean"].detach(),
        "scale": task_outputs["scale"].detach(),
    }
