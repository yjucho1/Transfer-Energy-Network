"""Training utilities for Transfer Energy Network."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .losses import correction_energy_nce_loss, gaussian_crps, gaussian_mean_quantile_loss, gaussian_nll


@dataclass
class TrainingConfig:
    """Configuration for a joint forecasting and transferability step."""

    forecast_loss_weight: float = 1.0
    base_forecast_loss_weight: float = 1.0
    forecast_loss_type: str = "nll"
    energy_temperature: float = 1.0
    energy_margin: float = 0.2
    energy_nce_temperature: float = 1.0
    energy_metric_temperature: float = 0.1
    trajectory_energy_weight: float = 0.1
    trajectory_energy_num_samples: int = 4


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


def _gaussian_crps_per_sample(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    scale = scale.clamp_min(1e-6)
    z = (target - mean) / scale
    pdf = torch.exp(-0.5 * z.pow(2)) / torch.sqrt(torch.tensor(2.0 * torch.pi, dtype=target.dtype, device=target.device))
    cdf = 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, dtype=target.dtype, device=target.device))))
    crps = scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / torch.sqrt(torch.tensor(torch.pi, dtype=target.dtype, device=target.device)))
    return crps.mean(dim=-1)


def _candidate_correction_energies_and_losses(
    model: torch.nn.Module,
    outputs: dict[str, torch.Tensor],
    forecast_target: torch.Tensor,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    candidate_energies = [outputs["correction_energy_pred"].unsqueeze(-1)]
    candidate_losses = [_gaussian_crps_per_sample(forecast_target, outputs["mean"], outputs["scale"]).unsqueeze(-1)]

    for _ in range(num_samples):
        eps = torch.randn_like(outputs["mean"])
        sampled_future = outputs["mean"] + outputs["scale"] * eps
        sampled_correction = sampled_future - outputs["base_mean"].detach()
        sampled_energy = model.score_correction_energy(
            target_repr=outputs["target_repr"],
            pooled_sources=outputs["pooled_sources"],
            correction=sampled_correction,
        )
        sampled_loss = _gaussian_crps_per_sample(forecast_target, sampled_future, outputs["scale"])
        candidate_energies.append(sampled_energy.unsqueeze(-1))
        candidate_losses.append(sampled_loss.unsqueeze(-1))

    return torch.cat(candidate_energies, dim=-1), torch.cat(candidate_losses, dim=-1)


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
    if isinstance(model.forecast_head, torch.nn.ModuleDict) and "residual" in model.forecast_head:
        forecast_task_module = model.forecast_head["residual"]
    else:
        forecast_task_module = model.forecast_head
    task_modules = [model.target_encoder, model.source_encoder, model.energy_mlp, forecast_task_module]

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
    candidate_energies, candidate_losses = _candidate_correction_energies_and_losses(
        model=model,
        outputs=loss_outputs,
        forecast_target=batch["forecast_target"],
        num_samples=config.trajectory_energy_num_samples,
    )
    best_indices = candidate_losses.argmin(dim=-1, keepdim=True)
    positive_energy = candidate_energies.gather(1, best_indices).squeeze(-1)
    negative_mask = torch.ones_like(candidate_energies, dtype=torch.bool)
    negative_mask.scatter_(1, best_indices, False)
    negative_energies_t = candidate_energies[negative_mask].view(candidate_energies.shape[0], -1)
    energy_loss = correction_energy_nce_loss(
        positive_energy=positive_energy,
        negative_energies=negative_energies_t,
        temperature=config.energy_nce_temperature,
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
    base_forecast_loss = compute_forecast_loss(
        target=batch["forecast_target"],
        mean=task_outputs["base_mean"],
        scale=task_outputs["base_scale"],
        loss_type=config.forecast_loss_type,
    )
    candidate_energies, candidate_losses = _candidate_correction_energies_and_losses(
        model=model,
        outputs=task_outputs,
        forecast_target=batch["forecast_target"],
        num_samples=config.trajectory_energy_num_samples,
    )
    metric_weights = torch.softmax(-candidate_losses / config.energy_metric_temperature, dim=-1)
    mc_energy = (metric_weights * candidate_energies).sum(dim=-1).mean()
    trajectory_energy_term = config.trajectory_energy_weight * mc_energy
    forecast_term = config.forecast_loss_weight * forecast_loss
    base_forecast_term = config.base_forecast_loss_weight * base_forecast_loss
    task_total = forecast_term + base_forecast_term + trajectory_energy_term
    task_total.backward()
    optimizer_task.step()

    for module in loss_modules:
        _set_requires_grad(module, True)

    return {
        "loss": (task_total + loss_term).detach(),
        "forecast_loss": forecast_loss.detach(),
        "base_forecast_loss": base_forecast_loss.detach(),
        "energy_loss": energy_loss.detach(),
        "trajectory_energy_loss": trajectory_energy_term.detach(),
        "weights": task_outputs["weights"].detach(),
        "energies": task_outputs["energies"].detach(),
        "base_mean": task_outputs["base_mean"].detach(),
        "base_scale": task_outputs["base_scale"].detach(),
        "residual_mean": task_outputs["residual_mean"].detach(),
        "correction_energy_base": task_outputs["correction_energy_base"].detach(),
        "correction_energy_pred": task_outputs["correction_energy_pred"].detach(),
        "correction_energy_gt": task_outputs["correction_energy_gt"].detach(),
        "mean": task_outputs["mean"].detach(),
        "scale": task_outputs["scale"].detach(),
    }
