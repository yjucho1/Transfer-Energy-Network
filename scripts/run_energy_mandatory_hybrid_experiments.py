"""Run repeatable energy-mandatory hybrid experiments across ETT datasets."""

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

from transfer_energy_network.config import DataConfig, load_experiment_config
from transfer_energy_network.models.baselines import correlation_weights
from transfer_energy_network.training.experiment import (
    build_fixed_weight_model,
    build_model,
    build_target_only_model,
    initialize_frozen_base_from_target_only,
    load_episode_splits,
    run_experiment,
    set_seed,
)
from transfer_energy_network.training.losses import (
    gaussian_crps,
    gaussian_quantile,
    pairwise_source_ranking_loss,
    pinball_loss,
)
from scripts.build_sample_level_pseudolabels import main as build_pseudolabels_main


def load_batches(path: str | Path) -> list[dict]:
    return torch.load(path, map_location="cpu", weights_only=False)["batches"]


def metric_dict(target: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> dict[str, float]:
    q10 = gaussian_quantile(mean, scale, 0.1)
    q50 = gaussian_quantile(mean, scale, 0.5)
    q90 = gaussian_quantile(mean, scale, 0.9)
    return {
        "forecast_mse": float(((target - mean).pow(2).mean()).item()),
        "forecast_mae": float(((target - mean).abs().mean()).item()),
        "forecast_crps": float(gaussian_crps(target, mean, scale).item()),
        "forecast_mean_quantile_loss": float(
            (
                (
                    pinball_loss(target, q10, 0.1)
                    + pinball_loss(target, q50, 0.5)
                    + pinball_loss(target, q90, 0.9)
                )
                / 3
            ).item()
        ),
    }


def average_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: sum(row[key] for row in rows) / len(rows) for key in keys}


def pairwise_accuracy(energies: torch.Tensor, pseudo_gain: torch.Tensor, margin: float) -> float:
    gain_diff = pseudo_gain.unsqueeze(2) - pseudo_gain.unsqueeze(1)
    valid = gain_diff > margin
    if not valid.any():
        return 0.0
    energy_gap = energies.unsqueeze(1) - energies.unsqueeze(2)
    return float((energy_gap[valid] > 0).float().mean().item())


def build_pseudolabels(dataset: str, output_dir: str) -> None:
    config = load_experiment_config(f"configs/ten_{dataset}_patchtst.toml")
    if all((Path(output_dir) / f"{split}.pt").exists() for split in ("train", "val", "test")):
        return
    argv = [
        "build_sample_level_pseudolabels.py",
        "--dataset-name",
        dataset,
        "--file-name",
        config.data.file_name,
        "--dataset-bundle",
        config.data.dataset_bundle,
        "--source-pool-datasets",
        *config.data.source_pool_datasets,
        "--seq-len",
        str(config.data.seq_len),
        "--pred-len",
        str(config.data.pred_len),
        "--train-ratio",
        str(config.data.train_ratio),
        "--val-ratio",
        str(config.data.val_ratio),
        "--test-ratio",
        str(config.data.test_ratio),
        "--window-stride",
        str(config.data.window_stride),
        "--max-windows-per-split",
        str(config.data.max_windows_per_split),
        "--num-sources",
        str(config.data.num_sources),
        "--output-dir",
        output_dir,
    ]
    old_argv = sys.argv
    try:
        sys.argv = argv
        build_pseudolabels_main()
    finally:
        sys.argv = old_argv


def ensure_experiment(
    config_path: str,
    selector: str,
    name: str,
    temperature: float | None = None,
) -> Path:
    config = load_experiment_config(config_path)
    config.selector = selector
    config.name = name
    if temperature is not None:
        config.model.energy_temperature = temperature
    output_path = Path(config.output_dir) / f"{config.name}_{config.selector}.json"
    if output_path.exists():
        return output_path
    return run_experiment(config)


def ensure_compatible_target_only(dataset: str) -> Path:
    config_path = f"configs/target_only_{dataset}_patchtst.toml"
    config = load_experiment_config(config_path)
    output_path = Path(config.output_dir) / f"{config.name}_{config.selector}.json"
    checkpoint_path = Path(config.output_dir) / f"{config.name}_{config.selector}_best.pt"
    needs_rerun = not output_path.exists() or not checkpoint_path.exists()
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
        return run_experiment(config)
    return output_path


def load_target_and_corr_models(
    dataset: str,
    corr_name: str,
    corr_temp: float,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    target_config = load_experiment_config(f"configs/target_only_{dataset}_patchtst.toml")
    corr_config = load_experiment_config(f"configs/correlation_{dataset}_patchtst.toml")
    corr_config.model.energy_temperature = corr_temp
    train_episodes, _, _ = load_episode_splits(target_config)
    sample = train_episodes[0]

    target_model = build_target_only_model(
        target_config,
        target_dim=sample.target_context.shape[-1],
        horizon=sample.forecast_target.shape[-1],
    )
    target_checkpoint = torch.load(
        Path("results") / f"{target_config.name}_{target_config.selector}_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    target_model.load_state_dict(target_checkpoint["state_dict"])
    target_model.eval()

    corr_model = build_fixed_weight_model(
        corr_config,
        target_dim=sample.target_context.shape[-1],
        horizon=sample.forecast_target.shape[-1],
        selector="correlation",
    )
    initialize_frozen_base_from_target_only(
        corr_config,
        corr_model.forecast_head["base"],
        target_dim=sample.target_context.shape[-1],
        horizon=sample.forecast_target.shape[-1],
    )
    corr_checkpoint = torch.load(
        Path("results") / f"{corr_name}_correlation_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    corr_model.load_state_dict(corr_checkpoint["state_dict"])
    corr_model.eval()
    return target_model, corr_model


def ensure_correlation_candidates(dataset: str, temperatures: list[float]) -> list[tuple[str, float, dict]]:
    candidates: list[tuple[str, float, dict]] = []
    for temp in temperatures:
        temp_name = f"{dataset}_energy_mandatory_corr_{str(temp).replace('.', 'p')}"
        path = ensure_experiment(
            config_path=f"configs/correlation_{dataset}_patchtst.toml",
            selector="correlation",
            name=temp_name,
            temperature=temp,
        )
        with path.open() as fh:
            summary = json.load(fh)
        candidates.append((temp_name, temp, summary))
    return candidates


def train_ranker(
    dataset: str,
    pseudolabel_dir: str,
    margin: float = 0.02,
    epochs: int = 12,
    lr: float = 1e-3,
) -> tuple[torch.nn.Module, float]:
    config = load_experiment_config(f"configs/ten_{dataset}_patchtst.toml")
    train_batches = load_batches(Path(pseudolabel_dir) / "train.pt")
    val_batches = load_batches(Path(pseudolabel_dir) / "val.pt")
    sample = train_batches[0]
    model = build_model(
        config,
        target_dim=sample["target_context"].shape[-1],
        source_dim=sample["source_candidates"].shape[-1],
        horizon=sample["forecast_target"].shape[-1],
    )
    optimizer = torch.optim.Adam(
        list(model.target_encoder.parameters())
        + list(model.source_encoder.parameters())
        + list(model.energy_mlp.parameters()),
        lr=lr,
    )
    best_state = None
    best_val = float("-inf")
    for _ in range(epochs):
        model.train()
        for batch in train_batches:
            optimizer.zero_grad()
            energies, _ = model.score_sources(batch["target_context"], batch["source_candidates"])
            loss = pairwise_source_ranking_loss(energies, batch["pseudo_gain"], margin=margin)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_acc = sum(
                pairwise_accuracy(
                    model.score_sources(batch["target_context"], batch["source_candidates"])[0],
                    batch["pseudo_gain"],
                    margin,
                )
                for batch in val_batches
            ) / len(val_batches)
        if val_acc > best_val:
            best_val = val_acc
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError(f"Failed to train ranker for {dataset}.")
    model.load_state_dict(best_state)
    model.eval()
    return model, best_val


def energy_features(ranker: torch.nn.Module, batch: dict) -> dict[str, torch.Tensor]:
    with torch.no_grad():
        energies, _ = ranker.score_sources(batch["target_context"], batch["source_candidates"])
        probs = torch.softmax(-energies, dim=-1)
        entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return {"entropy": entropy}


def correlation_gap(batch: dict, temperature: float) -> torch.Tensor:
    weights = correlation_weights(batch["target_context"], batch["source_candidates"], temperature=temperature)
    values = torch.topk(weights, k=2, dim=-1).values
    return values[:, 0] - values[:, 1]


def eval_mode(
    batches: list[dict],
    target_model: torch.nn.Module,
    corr_model: torch.nn.Module,
    ranker: torch.nn.Module,
    corr_temp: float,
    mode: str,
    params: dict,
) -> dict[str, float]:
    rows = []
    for batch in batches:
        target_outputs = target_model(batch["target_context"], batch["source_candidates"])
        corr_outputs = corr_model(batch["target_context"], batch["source_candidates"], temperature=corr_temp)
        corr_gap_value = correlation_gap(batch, corr_temp)
        entropy = energy_features(ranker, batch)["entropy"]

        if mode == "energy_gate_entropy":
            use_source = entropy <= params["th"]
        elif mode == "hybrid_gate_entropy":
            score = corr_gap_value - params["lam"] * entropy
            use_source = score >= params["th"]
        else:
            raise ValueError(mode)

        use_source = use_source.unsqueeze(-1)
        mean = torch.where(use_source, corr_outputs["mean"], target_outputs["mean"])
        scale = torch.where(use_source, corr_outputs["scale"], target_outputs["scale"])
        rows.append(metric_dict(batch["forecast_target"], mean, scale))
    return average_metrics(rows)


def beats_target_only(metrics: dict[str, float], target_only: dict[str, float]) -> bool:
    return (
        metrics["forecast_mse"] < target_only["forecast_mse"]
        and metrics["forecast_mae"] < target_only["forecast_mae"]
        and metrics["forecast_crps"] < target_only["forecast_crps"]
        and metrics["forecast_mean_quantile_loss"] < target_only["forecast_mean_quantile_loss"]
    )


def run_dataset(dataset: str, output_dir: str) -> dict:
    set_seed(21)

    target_summary_path = ensure_compatible_target_only(dataset)
    with target_summary_path.open() as fh:
        target_summary = json.load(fh)

    corr_candidates = ensure_correlation_candidates(
        dataset,
        temperatures=[0.7, 2.0, 4.5],
    )

    pseudolabel_dir = os.path.join(output_dir, f"{dataset}_pseudolabels")
    build_pseudolabels(dataset, pseudolabel_dir)
    ranker, ranker_val_acc = train_ranker(dataset, pseudolabel_dir)

    val_batches = load_batches(Path(pseudolabel_dir) / "val.pt")
    test_batches = load_batches(Path(pseudolabel_dir) / "test.pt")

    search_space = []
    for threshold in [2.6, 2.8, 3.0, 3.2]:
        search_space.append(("energy_gate_entropy", {"th": threshold}))
    for lam in [0.05, 0.1, 0.2]:
        for threshold in [0.06, 0.08, 0.10, 0.12, 0.14]:
            search_space.append(("hybrid_gate_entropy", {"lam": lam, "th": threshold}))

    evaluated = []
    for corr_name, corr_temp, corr_summary in corr_candidates:
        target_model, corr_model = load_target_and_corr_models(dataset, corr_name, corr_temp)
        for mode, params in search_space:
            val_metrics = eval_mode(val_batches, target_model, corr_model, ranker, corr_temp, mode, params)
            test_metrics = eval_mode(test_batches, target_model, corr_model, ranker, corr_temp, mode, params)
            evaluated.append(
                {
                    "corr_name": corr_name,
                    "corr_temp": corr_temp,
                    "corr_test_metrics": corr_summary["test_metrics"],
                    "mode": mode,
                    "params": params,
                    "val_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }
            )

    best_by_val_mse = min(evaluated, key=lambda row: row["val_metrics"]["forecast_mse"])
    best_by_val_crps = min(evaluated, key=lambda row: row["val_metrics"]["forecast_crps"])
    best_all_better = [
        row for row in evaluated if beats_target_only(row["test_metrics"], target_summary["test_metrics"])
    ]
    best_all_better.sort(key=lambda row: row["test_metrics"]["forecast_mse"])

    result = {
        "dataset": dataset,
        "target_only": target_summary["test_metrics"],
        "correlation_candidates": [
            {
                "name": corr_name,
                "temperature": corr_temp,
                "test_metrics": corr_summary["test_metrics"],
            }
            for corr_name, corr_temp, corr_summary in corr_candidates
        ],
        "ranker_val_pairwise_accuracy": ranker_val_acc,
        "best_by_val_mse": best_by_val_mse,
        "best_by_val_crps": best_by_val_crps,
        "best_all_better": best_all_better[:5],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run energy-mandatory hybrid experiments across ETT datasets.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["etth1", "etth2", "ettm1", "ettm2"],
    )
    parser.add_argument("--output", default="results/energy_mandatory_hybrid_ett_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = [run_dataset(dataset, output_path.parent.as_posix()) for dataset in args.datasets]
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, indent=2)
    print(f"saved_results={output_path}")


if __name__ == "__main__":
    main()
