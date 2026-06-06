"""Train a sample-level source ranker from probe pseudo-labels."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network.config import load_experiment_config
from transfer_energy_network.training.experiment import build_model, set_seed
from transfer_energy_network.training.losses import pairwise_source_ranking_loss


def load_batches(path: str | Path) -> list[dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload["batches"]


def pairwise_accuracy(energies: torch.Tensor, pseudo_gain: torch.Tensor, margin: float) -> tuple[float, int]:
    gain_diff = pseudo_gain.unsqueeze(2) - pseudo_gain.unsqueeze(1)
    valid = gain_diff > margin
    if not valid.any():
        return 0.0, 0
    energy_gap = energies.unsqueeze(1) - energies.unsqueeze(2)
    correct = (energy_gap[valid] > 0).float().mean()
    return float(correct.item()), int(valid.sum().item())


def evaluate_ranker(model, batches: list[dict], temperature: float, margin: float) -> dict[str, float]:
    model.eval()
    losses = []
    accuracies = []
    pair_counts = []
    selected_gains = []
    oracle_gains = []
    selected_positive = []
    with torch.no_grad():
        for batch in batches:
            energies, _ = model.score_sources(batch["target_context"], batch["source_candidates"])
            loss = pairwise_source_ranking_loss(energies, batch["pseudo_gain"], margin=margin)
            acc, pair_count = pairwise_accuracy(energies, batch["pseudo_gain"], margin=margin)
            selected_index = energies.argmin(dim=-1)
            selected_gain = batch["pseudo_gain"].gather(1, selected_index.unsqueeze(-1)).squeeze(-1)
            oracle_gain = batch["pseudo_gain"].max(dim=-1).values
            losses.append(float(loss.item()))
            if pair_count > 0:
                accuracies.append(acc)
                pair_counts.append(pair_count)
            selected_gains.append(float(selected_gain.mean().item()))
            oracle_gains.append(float(oracle_gain.mean().item()))
            selected_positive.append(float((selected_gain > 0).float().mean().item()))
    weighted_pair_acc = 0.0
    if pair_counts:
        weighted_pair_acc = sum(acc * count for acc, count in zip(accuracies, pair_counts)) / sum(pair_counts)
    return {
        "ranking_loss": sum(losses) / len(losses),
        "pairwise_accuracy": weighted_pair_acc,
        "mean_selected_gain": sum(selected_gains) / len(selected_gains),
        "mean_oracle_gain": sum(oracle_gains) / len(oracle_gains),
        "selected_positive_rate": sum(selected_positive) / len(selected_positive),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a sample-level source ranker.")
    parser.add_argument("--config", default="configs/ten_etth1_patchtst.toml")
    parser.add_argument("--pseudolabel-dir", default="results/pseudolabels/etth1_sample_level")
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output", default="results/etth1_sample_ranker.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    set_seed(config.seed)

    train_batches = load_batches(Path(args.pseudolabel_dir) / "train.pt")
    val_batches = load_batches(Path(args.pseudolabel_dir) / "val.pt")
    test_batches = load_batches(Path(args.pseudolabel_dir) / "test.pt")
    sample_batch = train_batches[0]
    model = build_model(
        config,
        target_dim=sample_batch["target_context"].shape[-1],
        source_dim=sample_batch["source_candidates"].shape[-1],
        horizon=sample_batch["forecast_target"].shape[-1],
    )

    optimizer = torch.optim.Adam(
        list(model.target_encoder.parameters())
        + list(model.source_encoder.parameters())
        + list(model.energy_mlp.parameters()),
        lr=args.lr,
    )

    history = []
    best_state = None
    best_val_acc = float("-inf")
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        epoch_accs = []
        epoch_pairs = []
        for batch in train_batches:
            optimizer.zero_grad()
            energies, _ = model.score_sources(batch["target_context"], batch["source_candidates"])
            loss = pairwise_source_ranking_loss(energies, batch["pseudo_gain"], margin=args.margin)
            loss.backward()
            optimizer.step()
            acc, pair_count = pairwise_accuracy(energies.detach(), batch["pseudo_gain"], margin=args.margin)
            epoch_losses.append(float(loss.item()))
            if pair_count > 0:
                epoch_accs.append(acc)
                epoch_pairs.append(pair_count)

        train_pair_acc = 0.0
        if epoch_pairs:
            train_pair_acc = sum(acc * count for acc, count in zip(epoch_accs, epoch_pairs)) / sum(epoch_pairs)
        val_metrics = evaluate_ranker(model, val_batches, temperature=config.model.energy_temperature, margin=args.margin)
        epoch_record = {
            "epoch": epoch,
            "train_ranking_loss": sum(epoch_losses) / len(epoch_losses),
            "train_pairwise_accuracy": train_pair_acc,
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(epoch_record)
        if val_metrics["pairwise_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["pairwise_accuracy"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No best state captured during sample-level ranker training.")

    model.load_state_dict(best_state)
    val_metrics = evaluate_ranker(model, val_batches, temperature=config.model.energy_temperature, margin=args.margin)
    test_metrics = evaluate_ranker(model, test_batches, temperature=config.model.energy_temperature, margin=args.margin)

    output = {
        "config": {
            "experiment_config": args.config,
            "pseudolabel_dir": args.pseudolabel_dir,
            "margin": args.margin,
            "epochs": args.epochs,
            "lr": args.lr,
        },
        "best_epoch": best_epoch,
        "best_val_pairwise_accuracy": best_val_acc,
        "history": history,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"saved_results={output_path}")


if __name__ == "__main__":
    main()
