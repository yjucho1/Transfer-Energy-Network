"""Target-only forecasting baselines."""

from __future__ import annotations

import torch
from torch import nn

from .forecasting import PatchTSTGaussianForecaster


class TargetOnlyPatchTSTModel(nn.Module):
    """PatchTST Gaussian forecaster that ignores source candidates."""

    def __init__(
        self,
        target_dim: int,
        seq_len: int,
        horizon: int,
        d_model: int,
        patch_len: int = 8,
        stride: int = 4,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = max(1, seq_len)
        self.target_channels = max(1, target_dim // self.seq_len) if target_dim % self.seq_len == 0 else 1
        self.forecaster = PatchTSTGaussianForecaster(
            input_channels=self.target_channels,
            context_length=self.seq_len,
            horizon=horizon,
            d_model=d_model,
            patch_len=patch_len,
            stride=stride,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            target_channel_index=self.target_channels - 1,
        )

    def forward(
        self,
        target_context: torch.Tensor,
        source_candidates: torch.Tensor | None = None,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        del source_candidates, temperature
        target_sequence = target_context.view(target_context.shape[0], self.seq_len, self.target_channels)
        mean, scale = self.forecaster(target_sequence)
        dummy_energies = torch.zeros(target_context.shape[0], 1, dtype=target_context.dtype, device=target_context.device)
        dummy_weights = torch.ones_like(dummy_energies)
        return {
            "energies": dummy_energies,
            "weights": dummy_weights,
            "mean": mean,
            "scale": scale,
        }
