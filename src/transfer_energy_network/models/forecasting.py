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


class RevIN(nn.Module):
    """Reversible instance normalization for time series inputs."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1, 1, num_features))
            self.affine_bias = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        self._cached_mean: torch.Tensor | None = None
        self._cached_std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._cached_mean = x.mean(dim=1, keepdim=True).detach()
            variance = x.var(dim=1, keepdim=True, unbiased=False)
            self._cached_std = torch.sqrt(variance + self.eps).detach()
            x = (x - self._cached_mean) / self._cached_std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x

        if mode == "denorm":
            if self._cached_mean is None or self._cached_std is None:
                raise RuntimeError("RevIN denorm called before norm.")
            if self.affine:
                x = (x - self.affine_bias) / self.affine_weight.clamp_min(self.eps)
            return x * self._cached_std + self._cached_mean

        raise ValueError(f"Unknown RevIN mode: {mode}")


class PatchTSTBackbone(nn.Module):
    """PatchTST encoder that returns per-variable patch features."""

    def __init__(
        self,
        n_vars: int,
        patch_num: int,
        patch_len: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_vars = n_vars
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.d_model = d_model
        self.patch_proj = nn.Linear(patch_len, d_model)
        self.positional = nn.Parameter(torch.zeros(1, 1, patch_num, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [bs, nvars, patch_len, patch_num]
        z = z.permute(0, 1, 3, 2)  # [bs, nvars, patch_num, patch_len]
        z = self.patch_proj(z) + self.positional[:, :, : z.shape[2]]
        batch_size, n_vars, patch_num, d_model = z.shape
        z = z.reshape(batch_size * n_vars, patch_num, d_model)
        z = self.encoder(z)
        z = z.reshape(batch_size, n_vars, patch_num, d_model)
        return z.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num]


class PatchTSTHead(nn.Module):
    """Per-variable horizon head followed by cross-variable aggregation."""

    def __init__(self, n_vars: int, d_model: int, patch_num: int, horizon: int) -> None:
        super().__init__()
        self.var_head = nn.Linear(d_model * patch_num, horizon)
        self.var_aggregator = nn.Linear(n_vars, 1)
        self.scale_head = nn.Linear(d_model * patch_num, horizon)
        self.scale_aggregator = nn.Linear(n_vars, 1)
        self.softplus = nn.Softplus()

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # z: [bs, nvars, d_model, patch_num]
        z = z.flatten(start_dim=2)
        per_var_mean = self.var_head(z)  # [bs, nvars, horizon]
        per_var_scale = self.softplus(self.scale_head(z)) + 1e-4

        mean = self.var_aggregator(per_var_mean.permute(0, 2, 1)).squeeze(-1)
        scale = self.scale_aggregator(per_var_scale.permute(0, 2, 1)).squeeze(-1)
        scale = self.softplus(scale) + 1e-4
        return mean, scale


class PatchTSTGaussianForecaster(nn.Module):
    """PatchTST-style Gaussian forecaster for horizon prediction."""

    def __init__(
        self,
        input_channels: int,
        context_length: int,
        horizon: int,
        d_model: int,
        patch_len: int = 8,
        stride: int = 4,
        n_heads: int = 2,
        n_layers: int = 2,
        d_ff: int = 64,
        dropout: float = 0.1,
        revin: bool = True,
        affine: bool = True,
        padding_patch: str = "end",
        target_channel_index: int = -1,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.context_length = context_length
        self.horizon = horizon
        self.revin = revin
        self.padding_patch = padding_patch
        self.target_channel_index = target_channel_index
        self.patch_len = min(patch_len, context_length)
        self.stride = max(1, min(stride, self.patch_len))
        self.patch_num = max(1, (context_length - self.patch_len) // self.stride + 1)
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
            self.patch_num += 1
        else:
            self.padding_patch_layer = None
        if revin:
            self.revin_layer = RevIN(input_channels, affine=affine)
        else:
            self.revin_layer = None
        self.backbone = PatchTSTBackbone(
            n_vars=input_channels,
            patch_num=self.patch_num,
            patch_len=self.patch_len,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.head = PatchTSTHead(
            n_vars=input_channels,
            d_model=d_model,
            patch_num=self.patch_num,
            horizon=horizon,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, context_length, channels]
        z = x.permute(0, 2, 1)  # [bs, nvars, seq_len]

        if self.revin and self.revin_layer is not None:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        if self.padding_patch == "end" and self.padding_patch_layer is not None:
            z = self.padding_patch_layer(z)

        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # [bs, nvars, patch_num, patch_len]
        z = z.permute(0, 1, 3, 2)  # [bs, nvars, patch_len, patch_num]

        z = self.backbone(z)  # [bs, nvars, d_model, patch_num]
        mean, scale = self.head(z)  # [bs, horizon], [bs, horizon]

        if self.revin and self.revin_layer is not None:
            target_index = self.target_channel_index
            if target_index < 0:
                target_index += self.input_channels
            target_mean = self.revin_layer._cached_mean[:, :, target_index : target_index + 1]
            target_std = self.revin_layer._cached_std[:, :, target_index : target_index + 1]
            mean = mean.unsqueeze(-1)
            if self.revin_layer.affine:
                mean = (mean - self.revin_layer.affine_bias[:, :, target_index : target_index + 1]) / (
                    self.revin_layer.affine_weight[:, :, target_index : target_index + 1].clamp_min(self.revin_layer.eps)
                )
            mean = (mean * target_std + target_mean).squeeze(-1)
            scale = scale * target_std.squeeze(-1)
        return mean, scale
