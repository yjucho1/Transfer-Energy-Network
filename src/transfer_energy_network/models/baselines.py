"""Source selection baselines for controlled comparisons."""

from __future__ import annotations

import torch
from torch import nn

from .forecasting import PatchTSTGaussianForecaster
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
    temperature: float = 1.0,
) -> torch.Tensor:
    del target_context
    del temperature
    num_sources = source_candidates.shape[1]
    return torch.full(
        (source_candidates.shape[0], num_sources),
        1.0 / num_sources,
        dtype=source_candidates.dtype,
        device=source_candidates.device,
    )
def ten_weights(energies: torch.Tensor, temperature: float) -> torch.Tensor:
    return energy_to_weights(energies, temperature=temperature)


class FixedWeightPatchTSTModel(nn.Module):
    """PatchTST forecaster with fixed source weighting and gated residual correction."""

    def __init__(
        self,
        seq_len: int,
        target_channels: int,
        horizon: int,
        d_model: int,
        weighting_fn,
        patch_len: int = 8,
        stride: int = 4,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.target_channels = target_channels
        self.weighting_fn = weighting_fn
        self.forecast_head = nn.ModuleDict(
            {
                "base": PatchTSTGaussianForecaster(
                    input_channels=target_channels,
                    context_length=seq_len,
                    horizon=horizon,
                    d_model=d_model,
                    patch_len=patch_len,
                    stride=stride,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                    target_channel_index=target_channels - 1,
                ),
                "residual": PatchTSTGaussianForecaster(
                    input_channels=target_channels + 1,
                    context_length=seq_len,
                    horizon=horizon,
                    d_model=d_model,
                    patch_len=patch_len,
                    stride=stride,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                    target_channel_index=target_channels - 1,
                ),
            }
        )
        self.source_residual_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(
        self,
        target_context: torch.Tensor,
        source_candidates: torch.Tensor,
        temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        weights = self.weighting_fn(
            target_context=target_context,
            source_candidates=source_candidates,
            temperature=temperature,
        )
        pooled_source_sequence = torch.sum(weights.unsqueeze(-1) * source_candidates, dim=1).unsqueeze(-1)
        target_sequence = target_context.view(target_context.shape[0], self.seq_len, self.target_channels)
        base_mean, base_scale = self.forecast_head["base"](target_sequence)
        forecast_input = torch.cat([target_sequence, pooled_source_sequence], dim=-1)
        residual_mean, _ = self.forecast_head["residual"](forecast_input)
        residual_gate = torch.sigmoid(self.source_residual_logit)
        mean = base_mean + residual_gate * residual_mean
        scale = base_scale
        dummy_energies = torch.zeros_like(weights)
        return {
            "energies": dummy_energies,
            "weights": weights,
            "mean": mean,
            "scale": scale,
            "source_residual_gate": residual_gate.detach(),
        }
