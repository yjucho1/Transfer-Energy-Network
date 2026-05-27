"""Probabilistic forecasting heads."""

from __future__ import annotations

import torch
from torch import nn


class GaussianForecastHead(nn.Module):
    """Predict mean and positive scale for a Gaussian forecast."""

    def __init__(self, input_dim: int, horizon: int) -> None:
        super().__init__()
        self.mean = nn.Linear(input_dim, horizon)
        self.scale = nn.Linear(input_dim, horizon)
        self.softplus = nn.Softplus()

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(features)
        scale = self.softplus(self.scale(features)) + 1e-4
        return mean, scale
