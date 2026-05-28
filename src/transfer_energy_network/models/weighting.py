"""Utilities for converting energies into transfer weights."""

from __future__ import annotations

import torch


def energy_to_weights(
    energies: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    """Convert energies into normalized transfer weights.

    Lower energy implies higher probability mass.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = -energies / temperature
    if top_k is None or top_k <= 0 or top_k >= logits.shape[-1]:
        return torch.softmax(logits, dim=-1)

    values, indices = torch.topk(logits, k=top_k, dim=-1)
    sparse_logits = torch.full_like(logits, float("-inf"))
    sparse_logits.scatter_(dim=-1, index=indices, src=values)
    return torch.softmax(sparse_logits, dim=-1)
