"""CLI entry point for experiment runs."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from transfer_energy_network.config import load_experiment_config
from transfer_energy_network.training.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TEN paper experiments.")
    parser.add_argument("config", help="Path to a TOML experiment config")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    output_path = run_experiment(config)
    print(f"saved_results={output_path}")


if __name__ == "__main__":
    main()
