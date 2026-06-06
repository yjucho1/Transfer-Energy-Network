"""Run fixed best energy-mandatory hybrid configs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from run_energy_mandatory_hybrid_experiments import (
    build_pseudolabels,
    ensure_compatible_target_only,
    ensure_experiment,
    eval_mode,
    load_batches,
    load_target_and_corr_models,
    train_ranker,
)


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def run_config(path: str | Path) -> dict:
    config = load_config(path)
    dataset = config["dataset"]
    output_dir = config.get("output_dir", "results")
    ensure_compatible_target_only(dataset)
    corr_temp = float(config["correlation"]["temperature"])
    corr_name = f"{dataset}_energy_mandatory_fixed_{str(corr_temp).replace('.', 'p')}"
    corr_path = ensure_experiment(
        config_path=f"configs/correlation_{dataset}_patchtst.toml",
        selector="correlation",
        name=corr_name,
        temperature=corr_temp,
    )
    with corr_path.open() as fh:
        corr_summary = json.load(fh)

    pseudolabel_dir = os.path.join(output_dir, f"{dataset}_pseudolabels")
    build_pseudolabels(dataset, pseudolabel_dir)
    ranker, ranker_val_acc = train_ranker(
        dataset,
        pseudolabel_dir,
        margin=float(config["ranker"].get("margin", 0.02)),
        epochs=int(config["ranker"].get("epochs", 12)),
        lr=float(config["ranker"].get("lr", 1e-3)),
    )
    target_model, corr_model = load_target_and_corr_models(dataset, corr_name, corr_temp)
    val_batches = load_batches(Path(pseudolabel_dir) / "val.pt")
    test_batches = load_batches(Path(pseudolabel_dir) / "test.pt")
    mode = config["hybrid"]["mode"]
    params = {key: value for key, value in config["hybrid"].items() if key != "mode"}
    val_metrics = eval_mode(val_batches, target_model, corr_model, ranker, corr_temp, mode, params)
    test_metrics = eval_mode(test_batches, target_model, corr_model, ranker, corr_temp, mode, params)
    return {
        "config_path": str(path),
        "name": config["name"],
        "dataset": dataset,
        "correlation_temperature": corr_temp,
        "correlation_test_metrics": corr_summary["test_metrics"],
        "ranker_val_pairwise_accuracy": ranker_val_acc,
        "mode": mode,
        "params": params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed best energy-mandatory configs.")
    parser.add_argument(
        "configs",
        nargs="*",
        default=[
            "configs/energy_mandatory_hybrid_etth1.toml",
            "configs/energy_mandatory_hybrid_etth2.toml",
            "configs/energy_mandatory_hybrid_ettm1.toml",
            "configs/energy_mandatory_hybrid_ettm2.toml",
        ],
    )
    parser.add_argument("--output", default="results/energy_mandatory_hybrid_best_configs.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = [run_config(path) for path in args.configs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, indent=2)
    print(f"saved_results={output_path}")


if __name__ == "__main__":
    main()
