"""Run fixed best lead-selective hybrid configs."""

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

from run_lead_selective_hybrid_ett_experiments import dataset_sweep, parse_args as _unused_parse_args


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def build_namespace(config: dict) -> argparse.Namespace:
    train = config["train"]
    selector = config["selector"]
    pred_len = int(config.get("pred_len", 96))
    return argparse.Namespace(
        datasets=[config["dataset"]],
        pred_lens=[pred_len],
        epochs=int(train.get("epochs", 8)),
        early_stopping_patience=int(train.get("early_stopping_patience", 3)),
        lr=float(train.get("lr", 1e-3)),
        proposer_hidden_dim=int(train.get("proposer_hidden_dim", 16)),
        proposer_dropout=float(train.get("proposer_dropout", 0.1)),
        gate_hidden_dim=int(train.get("gate_hidden_dim", 16)),
        proposer_temperature_sweep=[float(selector["proposer_temperature"])],
        gate_usage_penalty_sweep=[float(selector["gate_usage_penalty"])],
        gate_aux_weight_sweep=[float(selector.get("gate_aux_weight", 0.0))],
        gate_feature_mode_sweep=[str(selector["gate_feature_mode"])],
        source_pool_mode_sweep=[str(selector.get("source_pool_mode", "raw"))],
        scale_mode_sweep=[str(selector.get("scale_mode", "var_blend"))],
        gate_target_mode_sweep=[str(selector.get("gate_target_mode", "crps"))],
        hard_gate_inference=bool(selector.get("hard_gate_inference", False)),
        hard_gate_threshold=float(selector.get("hard_gate_threshold", 0.5)),
        output_json="",
        output_md="",
    )


def run_config(path: str | Path) -> dict:
    config = load_config(path)
    args = build_namespace(config)
    result = dataset_sweep(config["dataset"], int(config.get("pred_len", 96)), args)
    return {
        "config_path": str(path),
        "name": config["name"],
        "dataset": config["dataset"],
        "pred_len": int(config.get("pred_len", 96)),
        "result": result,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed best lead-selective configs.")
    parser.add_argument(
        "configs",
        nargs="*",
        default=[
            "configs/lead_selective_hybrid_etth1.toml",
            "configs/lead_selective_hybrid_etth2.toml",
            "configs/lead_selective_hybrid_ettm1.toml",
            "configs/lead_selective_hybrid_ettm2.toml",
        ],
    )
    parser.add_argument("--output", default="results/lead_selective_hybrid_best_configs.json")
    parser.add_argument("--pred-lens", nargs="+", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pred_lens:
        results = []
        for pred_len in args.pred_lens:
            for path in args.configs:
                config = load_config(path)
                config["pred_len"] = pred_len
                temp_path = Path("/private/tmp") / f"{Path(path).stem}_pred{pred_len}.toml"
                lines = [
                    f'name = "{config["name"]}_pred{pred_len}"',
                    f'dataset = "{config["dataset"]}"',
                    f"pred_len = {pred_len}",
                    f"seed = {config.get('seed', 21)}",
                    f'output_dir = "{config.get("output_dir", "results")}"',
                    "",
                    "[selector]",
                    f'proposer_temperature = {config["selector"]["proposer_temperature"]}',
                    f'gate_usage_penalty = {config["selector"]["gate_usage_penalty"]}',
                    f'gate_aux_weight = {config["selector"].get("gate_aux_weight", 0.0)}',
                    f'gate_feature_mode = "{config["selector"]["gate_feature_mode"]}"',
                    f'source_pool_mode = "{config["selector"].get("source_pool_mode", "raw")}"',
                    f'scale_mode = "{config["selector"].get("scale_mode", "var_blend")}"',
                    f'gate_target_mode = "{config["selector"].get("gate_target_mode", "crps")}"',
                    f'hard_gate_inference = {str(config["selector"].get("hard_gate_inference", False)).lower()}',
                    f'hard_gate_threshold = {config["selector"].get("hard_gate_threshold", 0.5)}',
                    "",
                    "[train]",
                    f'epochs = {config["train"].get("epochs", 8)}',
                    f'early_stopping_patience = {config["train"].get("early_stopping_patience", 3)}',
                    f'lr = {config["train"].get("lr", 1e-3)}',
                    f'proposer_hidden_dim = {config["train"].get("proposer_hidden_dim", 16)}',
                    f'proposer_dropout = {config["train"].get("proposer_dropout", 0.1)}',
                    f'gate_hidden_dim = {config["train"].get("gate_hidden_dim", 16)}',
                    "",
                ]
                temp_path.write_text("\n".join(lines), encoding="utf-8")
                results.append(run_config(temp_path))
    else:
        results = [run_config(path) for path in args.configs]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump({"results": results}, fh, indent=2)
    print(f"saved_results={output_path}")


if __name__ == "__main__":
    main()
