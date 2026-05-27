"""Source selection baselines for controlled comparisons."""

from __future__ import annotations

import torch

from .weighting import energy_to_weights


def correlation_weights(
    target_context: torch.Tensor,
    source_candidates: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """A simple similarity baseline using dot-product compatibility."""
    shared_dim = min(target_context.shape[-1], source_candidates.shape[-1])
    target = target_context[:, :shared_dim].unsqueeze(1)
    scores = (target * source_candidates[..., :shared_dim]).sum(dim=-1)
    return torch.softmax(scores / temperature, dim=-1)


def uniform_weights(
    target_context: torch.Tensor,
    source_candidates: torch.Tensor,
) -> torch.Tensor:
    del target_context
    num_sources = source_candidates.shape[1]
    return torch.full(
        (source_candidates.shape[0], num_sources),
        1.0 / num_sources,
        dtype=source_candidates.dtype,
        device=source_candidates.device,
    )


def oracle_weights(validation_delta: torch.Tensor) -> torch.Tensor:
    max_indices = validation_delta.argmax(dim=-1)
    weights = torch.zeros_like(validation_delta)
    weights.scatter_(1, max_indices.unsqueeze(1), 1.0)
    return weights


def ten_weights(energies: torch.Tensor, temperature: float) -> torch.Tensor:
    return energy_to_weights(energies, temperature=temperature)
