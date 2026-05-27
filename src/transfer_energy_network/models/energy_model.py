"""Core Transfer Energy Network modules."""

from __future__ import annotations

import torch
from torch import nn

from .forecasting import GaussianForecastHead
from .weighting import energy_to_weights


class TransferEnergyNetwork(nn.Module):
    """A small representation-level transferability model.

    Inputs:
    - target_context: [batch, target_dim]
    - source_candidates: [batch, num_sources, source_dim]
    """

    def __init__(
        self,
        target_dim: int,
        source_dim: int,
        hidden_dim: int,
        horizon: int,
    ) -> None:
        super().__init__()
        self.target_encoder = nn.Sequential(
            nn.Linear(target_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.source_encoder = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.energy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.forecast_head = GaussianForecastHead(input_dim=hidden_dim * 2, horizon=horizon)

    def encode_target(self, target_context: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(target_context)

    def encode_sources(self, source_candidates: torch.Tensor) -> torch.Tensor:
        return self.source_encoder(source_candidates)

    def score_sources(
        self,
        target_context: torch.Tensor,
        source_candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        target_repr = self.encode_target(target_context)
        source_repr = self.encode_sources(source_candidates)

        expanded_target = target_repr.unsqueeze(1).expand_as(source_repr)
        interaction = expanded_target * source_repr
        energy_inputs = torch.cat([expanded_target, source_repr, interaction], dim=-1)
        energies = self.energy_mlp(energy_inputs).squeeze(-1)
        return energies, source_repr

    def aggregate_sources(
        self,
        energies: torch.Tensor,
        source_repr: torch.Tensor,
        temperature: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = energy_to_weights(energies, temperature=temperature)
        pooled_sources = torch.sum(weights.unsqueeze(-1) * source_repr, dim=1)
        return pooled_sources, weights

    def forward(
        self,
        target_context: torch.Tensor,
        source_candidates: torch.Tensor,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        target_repr = self.encode_target(target_context)
        energies, source_repr = self.score_sources(target_context, source_candidates)
        pooled_sources, weights = self.aggregate_sources(
            energies=energies,
            source_repr=source_repr,
            temperature=temperature,
        )
        forecast_features = torch.cat([target_repr, pooled_sources], dim=-1)
        mean, scale = self.forecast_head(forecast_features)
        return {
            "energies": energies,
            "weights": weights,
            "mean": mean,
            "scale": scale,
        }
