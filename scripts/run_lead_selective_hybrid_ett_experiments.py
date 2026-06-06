"""Run repeatable lead-aware selective hybrid experiments across ETT datasets."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from copy import deepcopy

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network.config import ExperimentConfig, load_experiment_config
from transfer_energy_network.models import LeadSelectiveHybridModel
from transfer_energy_network.training.evaluation import EpochMetrics
from transfer_energy_network.training.experiment import (
    build_target_only_model,
    get_best_checkpoint_path,
    load_episode_splits,
    run_experiment,
    set_seed,
)
from transfer_energy_network.training.losses import gaussian_crps, gaussian_quantile, pinball_loss
from transfer_energy_network.training.trainer import compute_forecast_loss


def override_pred_len(config: ExperimentConfig, pred_len: int) -> ExperimentConfig:
    updated = deepcopy(config)
    updated.data.pred_len = pred_len
    updated.name = f"{config.name}_pred{pred_len}"
    return updated


def ensure_target_only_checkpoint(dataset: str, pred_len: int) -> tuple[ExperimentConfig, Path]:
    config = override_pred_len(
        load_experiment_config(f"configs/target_only_{dataset}_patchtst.toml"),
        pred_len=pred_len,
    )
    checkpoint_path = get_best_checkpoint_path(config)
    summary_path = Path(config.output_dir) / f"{config.name}_{config.selector}.json"
    needs_rerun = not checkpoint_path.exists() or not summary_path.exists()
    if not needs_rerun:
        train_episodes, _, _ = load_episode_splits(config)
        sample = train_episodes[0]
        model = build_target_only_model(
            config,
            target_dim=sample.target_context.shape[-1],
            horizon=sample.forecast_target.shape[-1],
        )
        try:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["state_dict"])
        except Exception:
            needs_rerun = True
    if needs_rerun:
        run_experiment(config)
    return config, checkpoint_path


def build_lead_model(
    config: ExperimentConfig,
    sample_batch,
    proposer_hidden_dim: int,
    proposer_temperature: float,
    proposer_dropout: float,
    gate_hidden_dim: int,
    gate_feature_mode: str,
) -> LeadSelectiveHybridModel:
    return LeadSelectiveHybridModel(
        seq_len=config.data.seq_len,
        horizon=sample_batch.forecast_target.shape[-1],
        hidden_dim=config.model.hidden_dim,
        proposer_hidden_dim=proposer_hidden_dim,
        proposer_temperature=proposer_temperature,
        proposer_dropout=proposer_dropout,
        gate_hidden_dim=gate_hidden_dim,
        gate_feature_mode=gate_feature_mode,
        patch_len=config.model.forecast_patch_len,
        stride=config.model.forecast_patch_stride,
        n_heads=config.model.forecast_n_heads,
        n_layers=config.model.forecast_n_layers,
        d_ff=config.model.forecast_d_ff,
        forecast_dropout=config.model.forecast_dropout,
    )


def load_frozen_base_into_hybrid(
    model: LeadSelectiveHybridModel,
    config: ExperimentConfig,
    checkpoint_path: Path,
    sample_batch,
) -> None:
    target_model = build_target_only_model(
        config,
        target_dim=sample_batch.target_context.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    target_model.load_state_dict(payload["state_dict"])
    model.load_frozen_base(target_model.forecaster.state_dict())


def compute_metrics(target: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> dict[str, float]:
    q10 = gaussian_quantile(mean, scale, 0.1)
    q50 = gaussian_quantile(mean, scale, 0.5)
    q90 = gaussian_quantile(mean, scale, 0.9)
    return {
        "forecast_mse": float(((target - mean).pow(2).mean()).item()),
        "forecast_mae": float(((target - mean).abs().mean()).item()),
        "forecast_crps": float(gaussian_crps(target, mean, scale).item()),
        "forecast_q10_loss": float(pinball_loss(target, q10, 0.1).item()),
        "forecast_q50_loss": float(pinball_loss(target, q50, 0.5).item()),
        "forecast_q90_loss": float(pinball_loss(target, q90, 0.9).item()),
        "forecast_mean_quantile_loss": float(
            (
                pinball_loss(target, q10, 0.1)
                + pinball_loss(target, q50, 0.5)
                + pinball_loss(target, q90, 0.9)
            ).div(3.0).item()
        ),
    }


def evaluate_hybrid(model: LeadSelectiveHybridModel, episodes: list, config: ExperimentConfig) -> EpochMetrics:
    rows: list[dict[str, float]] = []
    model.eval()
    with torch.no_grad():
        for batch in episodes:
            outputs = model(batch.target_context, batch.source_candidates)
            rows.append(compute_metrics(batch.forecast_target, outputs["mean"], outputs["scale"]))
    keys = rows[0].keys()
    averages = {key: sum(row[key] for row in rows) / len(rows) for key in keys}
    return EpochMetrics(**averages)


def baseline_metrics(config: ExperimentConfig, checkpoint_path: Path) -> tuple[dict[str, float], dict[str, float], list, list, list]:
    train_episodes, val_episodes, test_episodes = load_episode_splits(config)
    sample_batch = train_episodes[0]
    baseline_model = build_target_only_model(
        config,
        target_dim=sample_batch.target_context.shape[-1],
        horizon=sample_batch.forecast_target.shape[-1],
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    baseline_model.load_state_dict(payload["state_dict"])
    baseline_model.eval()

    val_rows: list[dict[str, float]] = []
    test_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in val_episodes:
            outputs = baseline_model(batch.target_context, batch.source_candidates)
            val_rows.append(compute_metrics(batch.forecast_target, outputs["mean"], outputs["scale"]))
        for batch in test_episodes:
            outputs = baseline_model(batch.target_context, batch.source_candidates)
            test_rows.append(compute_metrics(batch.forecast_target, outputs["mean"], outputs["scale"]))
    val_avg = {k: sum(row[k] for row in val_rows) / len(val_rows) for k in val_rows[0]}
    test_avg = {k: sum(row[k] for row in test_rows) / len(test_rows) for k in test_rows[0]}
    return val_avg, test_avg, train_episodes, val_episodes, test_episodes


def train_trial(
    dataset: str,
    target_config: ExperimentConfig,
    target_ckpt: Path,
    train_episodes: list,
    val_episodes: list,
    test_episodes: list,
    baseline_test_avg: dict[str, float],
    trial: dict[str, float | int | str],
    args: argparse.Namespace,
) -> dict:
    set_seed(target_config.seed)
    sample_batch = train_episodes[0]
    model = build_lead_model(
        target_config,
        sample_batch,
        proposer_hidden_dim=args.proposer_hidden_dim,
        proposer_temperature=float(trial["proposer_temperature"]),
        proposer_dropout=args.proposer_dropout,
        gate_hidden_dim=args.gate_hidden_dim,
        gate_feature_mode=str(trial["gate_feature_mode"]),
    )
    load_frozen_base_into_hybrid(model, target_config, target_ckpt, sample_batch)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
    )

    train_history: list[dict] = []
    validation_history: list[dict] = []
    best_epoch = 0
    best_val_mse = float("inf")
    best_state = None
    best_validation_metrics = None
    epochs_without_improvement = 0
    stopped_epoch = args.epochs

    trial_slug = (
        f"{dataset}_pred{target_config.data.pred_len}_lead_selective_"
        f"{str(trial['gate_feature_mode'])}_"
        f"temp{str(trial['proposer_temperature']).replace('.', 'p')}_"
        f"usage{str(trial['gate_usage_penalty']).replace('.', 'p')}_"
        f"aux{str(trial['gate_aux_weight']).replace('.', 'p')}"
    )
    checkpoint_path = Path("results") / f"{trial_slug}_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        forecast_losses: list[float] = []
        usage_terms: list[float] = []
        gate_losses: list[float] = []
        total_losses: list[float] = []

        for batch in train_episodes:
            optimizer.zero_grad()
            outputs = model(
                batch.target_context,
                batch.source_candidates,
                forecast_target=batch.forecast_target,
            )
            forecast_loss = compute_forecast_loss(
                target=batch.forecast_target,
                mean=outputs["mean"],
                scale=outputs["scale"],
                loss_type=target_config.optim.forecast_loss_type,
            )
            usage_penalty = float(trial["gate_usage_penalty"]) * outputs["gate"].mean()
            if outputs["gate_target"] is None or float(trial["gate_aux_weight"]) <= 0.0:
                gate_aux = torch.zeros_like(usage_penalty)
            else:
                gate_aux = float(trial["gate_aux_weight"]) * F.binary_cross_entropy(
                    outputs["gate"],
                    outputs["gate_target"],
                )
            loss = forecast_loss + usage_penalty + gate_aux
            loss.backward()
            optimizer.step()

            forecast_losses.append(float(forecast_loss.item()))
            usage_terms.append(float(usage_penalty.item()))
            gate_losses.append(float(gate_aux.item()))
            total_losses.append(float(loss.item()))

        train_history.append(
            {
                "epoch": epoch,
                "loss": sum(total_losses) / len(total_losses),
                "forecast_loss": sum(forecast_losses) / len(forecast_losses),
                "usage_penalty": sum(usage_terms) / len(usage_terms),
                "gate_aux_loss": sum(gate_losses) / len(gate_losses),
            }
        )

        val_metrics = asdict(evaluate_hybrid(model, val_episodes, target_config))
        val_metrics["epoch"] = epoch
        validation_history.append(val_metrics)

        if val_metrics["forecast_mse"] < best_val_mse - target_config.optim.early_stopping_min_delta:
            best_val_mse = val_metrics["forecast_mse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_validation_metrics = copy.deepcopy(val_metrics)
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": best_state,
                    "best_epoch": best_epoch,
                    "best_validation_metrics": best_validation_metrics,
                    "trial": trial,
                    "dataset": dataset,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.early_stopping_patience:
            stopped_epoch = epoch
            break

    if best_state is None or best_validation_metrics is None:
        raise RuntimeError(f"Training did not capture a best state for {dataset}.")

    model.load_state_dict(best_state)
    test_metrics = asdict(evaluate_hybrid(model, test_episodes, target_config))
    deltas = {
        key: test_metrics[key] - baseline_test_avg[key]
        for key in ("forecast_mse", "forecast_mae", "forecast_crps", "forecast_mean_quantile_loss")
    }
    all_better = all(value < 0.0 for value in deltas.values())
    return {
        "dataset": dataset,
        "trial": trial,
        "train_history": train_history,
        "validation_history": validation_history,
        "best_epoch": best_epoch,
        "best_validation_metrics": best_validation_metrics,
        "test_metrics": test_metrics,
        "test_delta_vs_target_only": deltas,
        "all_metrics_better_than_target_only": all_better,
        "best_checkpoint": str(checkpoint_path),
        "stopped_epoch": stopped_epoch,
    }


def select_best_trials(trials: list[dict]) -> tuple[dict, dict | None]:
    best_by_val = min(trials, key=lambda row: row["best_validation_metrics"]["forecast_mse"])
    all_better_trials = [row for row in trials if row["all_metrics_better_than_target_only"]]
    if not all_better_trials:
        return best_by_val, None
    best_all_better = min(all_better_trials, key=lambda row: row["best_validation_metrics"]["forecast_mse"])
    return best_by_val, best_all_better


def dataset_sweep(dataset: str, pred_len: int, args: argparse.Namespace) -> dict:
    target_config, target_ckpt = ensure_target_only_checkpoint(dataset, pred_len=pred_len)
    baseline_val_avg, baseline_test_avg, train_episodes, val_episodes, test_episodes = baseline_metrics(
        target_config,
        target_ckpt,
    )
    trials: list[dict] = []
    for proposer_temperature, gate_usage_penalty, gate_aux_weight, gate_feature_mode in itertools.product(
        args.proposer_temperature_sweep,
        args.gate_usage_penalty_sweep,
        args.gate_aux_weight_sweep,
        args.gate_feature_mode_sweep,
    ):
        trial = {
            "proposer_temperature": proposer_temperature,
            "gate_usage_penalty": gate_usage_penalty,
            "gate_aux_weight": gate_aux_weight,
            "gate_feature_mode": gate_feature_mode,
        }
        trials.append(
            train_trial(
                dataset=dataset,
                target_config=target_config,
                target_ckpt=target_ckpt,
                train_episodes=train_episodes,
                val_episodes=val_episodes,
                test_episodes=test_episodes,
                baseline_test_avg=baseline_test_avg,
                trial=trial,
                args=args,
            )
        )

    best_by_val, best_all_better = select_best_trials(trials)
    return {
        "dataset": dataset,
        "pred_len": pred_len,
        "baseline_validation_metrics": baseline_val_avg,
        "baseline_test_metrics": baseline_test_avg,
        "num_trials": len(trials),
        "best_by_validation_mse": best_by_val,
        "best_all_better": best_all_better,
        "trials": trials,
    }


def write_markdown(path: Path, dataset_results: list[dict]) -> None:
    lines = [
        "# Lead-Selective Hybrid Sweep Summary",
        "",
        "| Dataset | Pred | Best Config | MSE Delta | MAE Delta | CRPS Delta | Mean Q Delta | All Better |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in dataset_results:
        chosen = row["best_all_better"] or row["best_by_validation_mse"]
        delta = chosen["test_delta_vs_target_only"]
        trial = chosen["trial"]
        config_label = (
            f"`{trial['gate_feature_mode']}`, "
            f"`temp={trial['proposer_temperature']}`, "
            f"`usage={trial['gate_usage_penalty']}`, "
            f"`aux={trial['gate_aux_weight']}`"
        )
        lines.append(
            f"| `{row['dataset']}` | `{row['pred_len']}` | {config_label} | "
            f"{delta['forecast_mse']:+.6f} | "
            f"{delta['forecast_mae']:+.6f} | "
            f"{delta['forecast_crps']:+.6f} | "
            f"{delta['forecast_mean_quantile_loss']:+.6f} | "
            f"{'yes' if chosen['all_metrics_better_than_target_only'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["etth1", "etth2", "ettm1", "ettm2"])
    parser.add_argument("--pred-lens", nargs="+", type=int, default=[96])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--proposer-hidden-dim", type=int, default=16)
    parser.add_argument("--proposer-dropout", type=float, default=0.1)
    parser.add_argument("--gate-hidden-dim", type=int, default=16)
    parser.add_argument("--proposer-temperature-sweep", nargs="+", type=float, default=[1.0])
    parser.add_argument("--gate-usage-penalty-sweep", nargs="+", type=float, default=[0.005])
    parser.add_argument("--gate-aux-weight-sweep", nargs="+", type=float, default=[0.1])
    parser.add_argument("--gate-feature-mode-sweep", nargs="+", default=["basic"])
    parser.add_argument("--output-json", default="results/lead_selective_hybrid_ett_summary.json")
    parser.add_argument("--output-md", default="results/lead_selective_hybrid_ett_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_results = [
        dataset_sweep(dataset, pred_len, args)
        for pred_len in args.pred_lens
        for dataset in args.datasets
    ]
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(dataset_results, indent=2), encoding="utf-8")
    write_markdown(Path(args.output_md), dataset_results)
    print(json.dumps(dataset_results, indent=2))


if __name__ == "__main__":
    main()
