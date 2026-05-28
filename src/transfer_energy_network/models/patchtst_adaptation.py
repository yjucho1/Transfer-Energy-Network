"""Lightweight PatchTST-style model for adaptation-based source scoring."""

from __future__ import annotations

import torch
from torch import nn


class PatchTSTAdaptationModel(nn.Module):
    """A small patch-based transformer for one-step adaptation scoring.

    This is intentionally lightweight so we can fit it repeatedly while
    estimating per-batch transferability deltas.
    """

    def __init__(
        self,
        input_channels: int,
        context_length: int,
        patch_len: int = 8,
        stride: int = 4,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 1,
        d_ff: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.context_length = context_length
        self.patch_len = min(patch_len, context_length)
        self.stride = max(1, min(stride, self.patch_len))

        patch_num = max(1, (context_length - self.patch_len) // self.stride + 1)
        self.patch_proj = nn.Linear(self.patch_len * input_channels, d_model)
        self.positional = nn.Parameter(torch.zeros(1, patch_num, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model * patch_num),
            nn.Linear(d_model * patch_num, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, context_length, channels]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.contiguous().permute(0, 1, 3, 2)
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        tokens = self.patch_proj(patches) + self.positional[:, : patches.shape[1]]
        encoded = self.encoder(tokens)
        return self.head(encoded.reshape(encoded.shape[0], -1)).squeeze(-1)
