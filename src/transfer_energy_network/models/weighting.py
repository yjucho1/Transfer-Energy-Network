"""Utilities for converting energies into transfer weights."""

from __future__ import annotations

import torch


def energy_to_weights(energies: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Convert energies into normalized transfer weights.

    Lower energy implies higher probability mass.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.softmax(-energies / temperature, dim=-1)
