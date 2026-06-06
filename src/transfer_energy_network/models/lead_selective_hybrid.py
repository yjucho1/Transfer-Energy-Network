"""Lead-aware source proposer with a selective entropy gate."""

from __future__ import annotations

import torch
from torch import nn

from .forecasting import PatchTSTGaussianForecaster


def gaussian_crps_per_sample(
    target: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Closed-form Gaussian CRPS averaged over horizon, per sample."""
    scale = scale.clamp_min(1e-6)
    two = torch.tensor(2.0, dtype=target.dtype, device=target.device)
    pi = torch.tensor(torch.pi, dtype=target.dtype, device=target.device)
    z = (target - mean) / scale
    pdf = torch.exp(-0.5 * z.pow(2)) / torch.sqrt(two * pi)
    cdf = 0.5 * (1.0 + torch.erf(z / torch.sqrt(two)))
    crps = scale * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / torch.sqrt(pi))
    return crps.mean(dim=-1)


class RelativeLagBias(nn.Module):
    """Learned relative lag bias for lead-lag-aware attention."""

    def __init__(self, seq_len: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.bias = nn.Parameter(torch.zeros(2 * seq_len - 1))

    def forward(self, device: torch.device) -> torch.Tensor:
        positions = torch.arange(self.seq_len, device=device)
        rel = positions[:, None] - positions[None, :]
        rel = rel + (self.seq_len - 1)
        return self.bias[rel]


class LeadAwareCrossAttention(nn.Module):
    """Single-head cross-attention with learned relative lag bias."""

    def __init__(self, hidden_dim: int, dropout: float, seq_len: int) -> None:
        super().__init__()
        self.scale = hidden_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.rel_bias = RelativeLagBias(seq_len)

    def forward(self, target_embed: torch.Tensor, source_embed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(target_embed)
        k = self.k_proj(source_embed)
        v = self.v_proj(source_embed)
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_logits = attn_logits + self.rel_bias(target_embed.device).unsqueeze(0)
        attn = torch.softmax(attn_logits, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return self.out_proj(out), attn


class LeadAwareSourceProposer(nn.Module):
    """Target-conditioned cross-attention proposer over source candidates."""

    def __init__(
        self,
        seq_len: int,
        hidden_dim: int,
        temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.temperature = temperature
        self.target_proj = nn.Linear(1, hidden_dim)
        self.source_proj = nn.Linear(1, hidden_dim)
        self.cross_attn = LeadAwareCrossAttention(hidden_dim=hidden_dim, dropout=dropout, seq_len=seq_len)
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, target_context: torch.Tensor, source_candidates: torch.Tensor) -> dict[str, torch.Tensor]:
        _, num_sources, _ = source_candidates.shape
        target_embed = self.target_proj(target_context.unsqueeze(-1))
        source_embed = self.source_proj(source_candidates.unsqueeze(-1))

        attended = []
        attn_maps = []
        for source_index in range(num_sources):
            out, attn = self.cross_attn(target_embed, source_embed[:, source_index])
            attended.append(out)
            attn_maps.append(attn)

        attended_t = torch.stack(attended, dim=1)
        attn_maps_t = torch.stack(attn_maps, dim=1)

        target_summary = target_embed.mean(dim=1).unsqueeze(1).expand(-1, num_sources, -1)
        source_summary = attended_t.mean(dim=2)
        interaction = target_summary * source_summary
        score_input = torch.cat([target_summary, source_summary, interaction], dim=-1)
        scores = self.score_mlp(score_input).squeeze(-1)
        weights = torch.softmax(scores / self.temperature, dim=-1)

        entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=-1)
        top2 = torch.topk(weights, k=min(2, weights.shape[-1]), dim=-1).values
        if top2.shape[-1] > 1:
            gap = top2[:, 0] - top2[:, 1]
        else:
            gap = torch.zeros_like(top2[:, 0])
        return {
            "scores": scores,
            "weights": weights,
            "entropy": entropy,
            "gap": gap,
            "attn_maps": attn_maps_t,
            "attended_states": attended_t,
        }


class SelectiveGate(nn.Module):
    """Small gate network for source-aware forecast usage."""

    def __init__(self, input_dim: int = 3, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.mlp(features)).squeeze(-1)


class LeadSelectiveHybridModel(nn.Module):
    """Lead-aware proposer + source-aware forecaster + entropy gate."""

    def __init__(
        self,
        seq_len: int,
        horizon: int,
        hidden_dim: int,
        proposer_hidden_dim: int = 16,
        proposer_temperature: float = 1.0,
        proposer_dropout: float = 0.1,
        gate_hidden_dim: int = 16,
        gate_feature_mode: str = "basic",
        source_pool_mode: str = "raw",
        scale_mode: str = "var_blend",
        gate_target_mode: str = "crps",
        hard_gate_inference: bool = False,
        hard_gate_threshold: float = 0.5,
        patch_len: int = 8,
        stride: int = 4,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 64,
        forecast_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.horizon = horizon
        self.gate_feature_mode = gate_feature_mode
        self.source_pool_mode = source_pool_mode
        self.scale_mode = scale_mode
        self.gate_target_mode = gate_target_mode
        self.hard_gate_inference = hard_gate_inference
        self.hard_gate_threshold = hard_gate_threshold
        self.base_forecaster = PatchTSTGaussianForecaster(
            input_channels=1,
            context_length=seq_len,
            horizon=horizon,
            d_model=hidden_dim,
            patch_len=patch_len,
            stride=stride,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=forecast_dropout,
            target_channel_index=0,
        )
        self.source_forecaster = PatchTSTGaussianForecaster(
            input_channels=2,
            context_length=seq_len,
            horizon=horizon,
            d_model=hidden_dim,
            patch_len=patch_len,
            stride=stride,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=forecast_dropout,
            target_channel_index=0,
        )
        self.proposer = LeadAwareSourceProposer(
            seq_len=seq_len,
            hidden_dim=proposer_hidden_dim,
            temperature=proposer_temperature,
            dropout=proposer_dropout,
        )
        self.attended_pool_proj = nn.Linear(proposer_hidden_dim, 1)
        gate_input_dim = 6 if gate_feature_mode == "basic" else 7
        self.gate = SelectiveGate(input_dim=gate_input_dim, hidden_dim=gate_hidden_dim)

    def load_frozen_base(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.base_forecaster.load_state_dict(state_dict)
        for parameter in self.base_forecaster.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        target_context: torch.Tensor,
        source_candidates: torch.Tensor,
        forecast_target: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        proposer_out = self.proposer(target_context, source_candidates)
        weights = proposer_out["weights"]

        target_sequence = target_context.unsqueeze(-1)
        if self.source_pool_mode == "raw":
            pooled_source = torch.sum(weights.unsqueeze(-1) * source_candidates, dim=1)
            source_sequence = pooled_source.unsqueeze(-1)
        elif self.source_pool_mode == "attended":
            pooled_attended = torch.sum(
                weights.unsqueeze(-1).unsqueeze(-1) * proposer_out["attended_states"],
                dim=1,
            )
            source_sequence = self.attended_pool_proj(pooled_attended)
            pooled_source = source_sequence.squeeze(-1)
        else:
            raise ValueError(f"Unsupported source_pool_mode: {self.source_pool_mode}")
        base_mean, base_scale = self.base_forecaster(target_sequence)
        source_mean, source_scale = self.source_forecaster(torch.cat([target_sequence, source_sequence], dim=-1))

        pred_diff_norm = (source_mean - base_mean).pow(2).mean(dim=-1).sqrt()
        base_scale_mean = base_scale.mean(dim=-1)
        source_scale_mean = source_scale.mean(dim=-1)
        scale_ratio = source_scale_mean / base_scale_mean.clamp_min(1e-6)
        gate_feature_columns = [
            proposer_out["entropy"],
            proposer_out["gap"],
            pred_diff_norm,
            base_scale_mean,
            source_scale_mean,
            scale_ratio,
        ]
        attn_entropy = None
        if self.gate_feature_mode == "attn":
            mean_attn = proposer_out["attn_maps"].mean(dim=2)
            source_attn_entropy = -(mean_attn.clamp_min(1e-8) * mean_attn.clamp_min(1e-8).log()).sum(dim=-1)
            attn_entropy = (proposer_out["weights"] * source_attn_entropy).sum(dim=-1)
            gate_feature_columns.append(attn_entropy)
        gate_features = torch.stack(gate_feature_columns, dim=-1)
        gate = self.gate(gate_features)
        if (not self.training) and self.hard_gate_inference:
            gate_used = (gate >= self.hard_gate_threshold).float()
        else:
            gate_used = gate
        gate_expanded = gate_used.unsqueeze(-1)
        final_mean = (1.0 - gate_expanded) * base_mean + gate_expanded * source_mean
        if self.scale_mode == "base":
            final_scale = base_scale
        elif self.scale_mode == "var_blend":
            base_var = base_scale.pow(2)
            source_var = source_scale.pow(2)
            final_var = (1.0 - gate_expanded) * base_var + gate_expanded * source_var
            final_scale = final_var.clamp_min(1e-6).sqrt()
        elif self.scale_mode == "moment_match":
            second_moment = (
                (1.0 - gate_expanded) * (base_scale.pow(2) + base_mean.pow(2))
                + gate_expanded * (source_scale.pow(2) + source_mean.pow(2))
            )
            final_var = (second_moment - final_mean.pow(2)).clamp_min(1e-6)
            final_scale = final_var.sqrt()
        else:
            raise ValueError(f"Unsupported scale_mode: {self.scale_mode}")

        if forecast_target is None:
            gate_target = None
        else:
            if self.gate_target_mode == "mse":
                base_loss = (forecast_target - base_mean).pow(2).mean(dim=-1)
                source_loss = (forecast_target - source_mean).pow(2).mean(dim=-1)
            elif self.gate_target_mode == "crps":
                base_loss = gaussian_crps_per_sample(forecast_target, base_mean, base_scale)
                source_loss = gaussian_crps_per_sample(forecast_target, source_mean, source_scale)
            else:
                raise ValueError(f"Unsupported gate_target_mode: {self.gate_target_mode}")
            gate_target = (source_loss < base_loss).float()

        return {
            "mean": final_mean,
            "scale": final_scale,
            "base_mean": base_mean,
            "base_scale": base_scale,
            "source_mean": source_mean,
            "source_scale": source_scale,
            "weights": weights,
            "pooled_source": pooled_source,
            "proposer_scores": proposer_out["scores"],
            "proposer_entropy": proposer_out["entropy"],
            "proposer_gap": proposer_out["gap"],
            "attention_entropy": attn_entropy,
            "gate": gate,
            "gate_used": gate_used,
            "gate_target": gate_target,
            "pred_diff_norm": pred_diff_norm,
            "base_scale_mean": base_scale_mean,
            "source_scale_mean": source_scale_mean,
            "scale_ratio": scale_ratio,
        }
