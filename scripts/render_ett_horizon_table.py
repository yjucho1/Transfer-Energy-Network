"""Render horizon-specific ETT benchmark tables from JSON results."""

from __future__ import annotations

import json
import math
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2")
DATASET_KEYS = ("etth1", "etth2", "ettm1", "ettm2")
HORIZONS = (96, 192, 336, 720)
METHODS = (
    ("TEN", "ten"),
    ("Correlation", "correlation"),
    ("Uniform", "uniform"),
)
DEFAULT_SEEDS = (21, 22, 23, 24, 25)


def read_metrics(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload["test_metrics"]


def summarize(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.6f}"
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    std = math.sqrt(variance)
    return f"{mean:.6f} ± {std:.6f}"


def collect_metric_values(dataset_key: str, horizon: int, method_key: str, metric_name: str) -> list[float]:
    values: list[float] = []

    for seed in DEFAULT_SEEDS:
        path = Path("results") / f"{dataset_key}_h{horizon}_s{seed}_patchtst_transferability_{method_key}.json"
        if path.exists():
            values.append(read_metrics(path)[metric_name])

    if values:
        return values

    fallback_path = Path("results") / f"{dataset_key}_h{horizon}_patchtst_transferability_{method_key}.json"
    if fallback_path.exists():
        return [read_metrics(fallback_path)[metric_name]]

    raise FileNotFoundError(
        f"Missing results for dataset={dataset_key}, horizon={horizon}, method={method_key}"
    )


def main() -> None:
    lines = [
        "| Dataset | Horizon | Method | MSE | MAE | CRPS | Mean Q |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]

    for dataset, dataset_key in zip(DATASETS, DATASET_KEYS):
        for horizon in HORIZONS:
            for method_label, method_key in METHODS:
                lines.append(
                    "| {dataset} | {horizon} | {method} | {mse} | {mae} | {crps} | {mean_q} |".format(
                        dataset=dataset,
                        horizon=horizon,
                        method=method_label,
                        mse=summarize(collect_metric_values(dataset_key, horizon, method_key, "forecast_mse")),
                        mae=summarize(collect_metric_values(dataset_key, horizon, method_key, "forecast_mae")),
                        crps=summarize(collect_metric_values(dataset_key, horizon, method_key, "forecast_crps")),
                        mean_q=summarize(
                            collect_metric_values(dataset_key, horizon, method_key, "forecast_mean_quantile_loss")
                        ),
                    )
                )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
