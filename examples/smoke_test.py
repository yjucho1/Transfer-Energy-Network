"""Smoke test for the Transfer Energy Network scaffold."""

from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network import TransferEnergyNetwork


def main() -> None:
    torch.manual_seed(7)

    batch_size = 4
    num_sources = 5
    target_dim = 12
    source_dim = 8
    hidden_dim = 16
    horizon = 3

    model = TransferEnergyNetwork(
        target_dim=target_dim,
        source_dim=source_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
    )

    target_context = torch.randn(batch_size, target_dim)
    source_candidates = torch.randn(batch_size, num_sources, source_dim)
    outputs = model(target_context, source_candidates, temperature=0.7)

    print("energies shape:", tuple(outputs["energies"].shape))
    print("weights row sums:", outputs["weights"].sum(dim=-1))
    print("forecast mean shape:", tuple(outputs["mean"].shape))
    print("forecast scale min:", float(outputs["scale"].min()))


if __name__ == "__main__":
    main()
