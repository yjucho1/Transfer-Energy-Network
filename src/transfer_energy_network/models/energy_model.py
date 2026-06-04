"""Core Transfer Energy Network modules."""

from __future__ import annotations

import torch
from torch import nn

from .forecasting import GaussianForecastHead, PatchTSTGaussianForecaster
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
        seq_len: int,
        source_pooling: str = "energy",
        forecast_backbone: str = "patchtst",
        forecast_patch_len: int = 8,
        forecast_patch_stride: int = 4,
        forecast_n_heads: int = 2,
        forecast_n_layers: int = 2,
        forecast_d_ff: int = 64,
        forecast_dropout: float = 0.1,
        energy_top_k: int = 0,
        energy_logit_clip: float = 0.0,
        energy_center_logits: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = max(1, seq_len)
        self.source_pooling = source_pooling
        self.energy_top_k = energy_top_k
        self.energy_logit_clip = energy_logit_clip
        self.energy_center_logits = energy_center_logits
        self.target_channels = max(1, target_dim // self.seq_len) if target_dim % self.seq_len == 0 else 1
        self.use_patchtst_forecaster = (
            forecast_backbone == "patchtst"
            and target_dim % self.seq_len == 0
            and source_dim == self.seq_len
        )
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
        self.future_encoder = nn.Sequential(
            nn.Linear(horizon, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.trajectory_energy_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        if self.use_patchtst_forecaster:
            self.forecast_head = nn.ModuleDict(
                {
                    "base": PatchTSTGaussianForecaster(
                        input_channels=self.target_channels,
                        context_length=self.seq_len,
                        horizon=horizon,
                        d_model=hidden_dim,
                        patch_len=forecast_patch_len,
                        stride=forecast_patch_stride,
                        n_heads=forecast_n_heads,
                        n_layers=forecast_n_layers,
                        d_ff=forecast_d_ff,
                        dropout=forecast_dropout,
                        target_channel_index=self.target_channels - 1,
                    ),
                    "residual": PatchTSTGaussianForecaster(
                        input_channels=self.target_channels + 1,
                        context_length=self.seq_len,
                        horizon=horizon,
                        d_model=hidden_dim,
                        patch_len=forecast_patch_len,
                        stride=forecast_patch_stride,
                        n_heads=forecast_n_heads,
                        n_layers=forecast_n_layers,
                        d_ff=forecast_d_ff,
                        dropout=forecast_dropout,
                        target_channel_index=self.target_channels - 1,
                    ),
                }
            )
            self.source_residual_logit = nn.Parameter(torch.tensor(-4.0))
        else:
            self.forecast_head = nn.ModuleDict(
                {
                    "base": GaussianForecastHead(input_dim=hidden_dim, horizon=horizon),
                    "residual": GaussianForecastHead(input_dim=hidden_dim * 2, horizon=horizon),
                }
            )
            self.source_residual_logit = nn.Parameter(torch.tensor(-4.0))

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
        if self.source_pooling == "average":
            num_sources = source_repr.shape[1]
            weights = torch.full(
                (source_repr.shape[0], num_sources),
                1.0 / num_sources,
                dtype=source_repr.dtype,
                device=source_repr.device,
            )
            pooled_sources = source_repr.mean(dim=1)
            return pooled_sources, weights
        weights = energy_to_weights(
            energies,
            temperature=temperature,
            top_k=self.energy_top_k,
            logit_clip=self.energy_logit_clip,
            center_logits=self.energy_center_logits,
        )
        pooled_sources = torch.sum(weights.unsqueeze(-1) * source_repr, dim=1)
        return pooled_sources, weights

    def score_correction_energy(
        self,
        target_repr: torch.Tensor,
        pooled_sources: torch.Tensor,
        correction: torch.Tensor,
    ) -> torch.Tensor:
        correction_repr = self.future_encoder(correction)
        target_future_interaction = target_repr * correction_repr
        source_future_interaction = pooled_sources * correction_repr
        energy_inputs = torch.cat(
            [
                target_repr,
                pooled_sources,
                correction_repr,
                target_future_interaction,
                source_future_interaction,
            ],
            dim=-1,
        )
        return self.trajectory_energy_mlp(energy_inputs).squeeze(-1)

    def forward(
        self,
        target_context: torch.Tensor,
        source_candidates: torch.Tensor,
        temperature: float = 1.0,
        forecast_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        target_repr = self.encode_target(target_context)
        if self.source_pooling == "average":
            source_repr = self.encode_sources(source_candidates)
            energies = torch.zeros(
                source_candidates.shape[0],
                source_candidates.shape[1],
                dtype=source_candidates.dtype,
                device=source_candidates.device,
            )
        else:
            energies, source_repr = self.score_sources(target_context, source_candidates)
        pooled_sources, weights = self.aggregate_sources(
            energies=energies,
            source_repr=source_repr,
            temperature=temperature,
        )
        if self.use_patchtst_forecaster:
            target_sequence = target_context.view(target_context.shape[0], self.seq_len, self.target_channels)
            base_mean, base_scale = self.forecast_head["base"](target_sequence)
            if self.source_pooling == "average":
                pooled_source_sequence = source_candidates.mean(dim=1, keepdim=False).unsqueeze(-1)
            else:
                pooled_source_sequence = torch.sum(
                    weights.unsqueeze(-1) * source_candidates,
                    dim=1,
                ).unsqueeze(-1)
            forecast_input = torch.cat([target_sequence, pooled_source_sequence], dim=-1)
            residual_mean, _ = self.forecast_head["residual"](forecast_input)
            residual_gate = torch.sigmoid(self.source_residual_logit)
            gated_residual = residual_gate * residual_mean
        else:
            base_mean, base_scale = self.forecast_head["base"](target_repr)
            forecast_features = torch.cat([target_repr, pooled_sources], dim=-1)
            residual_mean, _ = self.forecast_head["residual"](forecast_features)
            residual_gate = torch.sigmoid(self.source_residual_logit)
            gated_residual = residual_gate * residual_mean
        detached_base_mean = base_mean.detach()
        detached_base_scale = base_scale.detach()
        # Keep the base forecaster aligned purely with the base forecasting loss.
        # Final forecast supervision and energy supervision only see detached
        # base predictions plus the source-induced correction.
        mean = detached_base_mean + gated_residual
        scale = detached_base_scale
        zero_correction = torch.zeros_like(base_mean)
        correction_energy_base = self.score_correction_energy(
            target_repr,
            torch.zeros_like(pooled_sources),
            zero_correction,
        )
        correction_energy_pred = self.score_correction_energy(target_repr, pooled_sources, gated_residual)
        if forecast_target is None:
            correction_energy_gt = None
        else:
            target_correction = forecast_target - detached_base_mean
            correction_energy_gt = self.score_correction_energy(target_repr, pooled_sources, target_correction)
        return {
            "energies": energies,
            "weights": weights,
            "target_repr": target_repr,
            "pooled_sources": pooled_sources,
            "base_mean": base_mean,
            "base_scale": base_scale,
            "residual_mean": gated_residual,
            "mean": mean,
            "scale": scale,
            "correction_energy_base": correction_energy_base,
            "source_residual_gate": torch.sigmoid(self.source_residual_logit).detach(),
            "residual_norm": gated_residual.pow(2).mean(),
            "correction_energy_pred": correction_energy_pred,
            "correction_energy_gt": correction_energy_gt,
        }
