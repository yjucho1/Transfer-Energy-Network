"""Synthetic training example for validation-delta supervised TEN."""

from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network import TrainingConfig, TransferEnergyNetwork, train_step


def make_synthetic_batch(
    batch_size: int,
    num_sources: int,
    target_dim: int,
    source_dim: int,
    horizon: int,
) -> dict[str, torch.Tensor]:
    """Create a toy episode where one source is more useful than the others."""
    target_context = torch.randn(batch_size, target_dim)
    source_candidates = torch.randn(batch_size, num_sources, source_dim)

    useful_source_index = torch.randint(0, num_sources, (batch_size,))
    validation_delta = torch.full((batch_size, num_sources), -0.5)

    for row in range(batch_size):
        chosen = useful_source_index[row]
        source_candidates[row, chosen, : min(target_dim, source_dim)] += target_context[
            row, : min(target_dim, source_dim)
        ]
        validation_delta[row, chosen] = 1.5

    forecast_signal = source_candidates[
        torch.arange(batch_size), useful_source_index, :horizon
    ]
    forecast_target = 0.6 * target_context[:, :horizon] + 0.4 * forecast_signal

    return {
        "target_context": target_context,
        "source_candidates": source_candidates,
        "forecast_target": forecast_target,
        "validation_delta": validation_delta,
    }


def main() -> None:
    torch.manual_seed(21)

    batch_size = 16
    num_sources = 6
    target_dim = 10
    source_dim = 10
    hidden_dim = 24
    horizon = 4

    model = TransferEnergyNetwork(
        target_dim=target_dim,
        source_dim=source_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    config = TrainingConfig(
        forecast_loss_weight=1.0,
        transfer_loss_weight=0.5,
        energy_temperature=0.7,
        target_temperature=0.7,
    )

    for step in range(1, 6):
        batch = make_synthetic_batch(
            batch_size=batch_size,
            num_sources=num_sources,
            target_dim=target_dim,
            source_dim=source_dim,
            horizon=horizon,
        )
        metrics = train_step(model, optimizer, batch, config)
        alignment = (
            metrics["weights"].argmax(dim=-1) == metrics["target_weights"].argmax(dim=-1)
        ).float()
        print(
            f"step={step} "
            f"loss={metrics['loss'].item():.4f} "
            f"forecast={metrics['forecast_loss'].item():.4f} "
            f"transfer={metrics['transfer_loss'].item():.4f} "
            f"top1_align={alignment.mean().item():.3f}"
        )


if __name__ == "__main__":
    main()
