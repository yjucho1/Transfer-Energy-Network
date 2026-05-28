"""Render a compact markdown table from ETT benchmark JSON results."""

from __future__ import annotations

import json
from pathlib import Path


RESULT_SPECS = [
    ("ETTh1", "results/etth1_patchtst_transferability_ten.json", "TEN"),
    ("ETTh1", "results/etth1_patchtst_transferability_correlation.json", "Correlation"),
    ("ETTh1", "results/etth1_patchtst_transferability_uniform.json", "Uniform"),
    ("ETTh2", "results/etth2_patchtst_transferability_ten.json", "TEN"),
    ("ETTh2", "results/etth2_patchtst_transferability_correlation.json", "Correlation"),
    ("ETTh2", "results/etth2_patchtst_transferability_uniform.json", "Uniform"),
    ("ETTm1", "results/ettm1_patchtst_transferability_ten.json", "TEN"),
    ("ETTm1", "results/ettm1_patchtst_transferability_correlation.json", "Correlation"),
    ("ETTm1", "results/ettm1_patchtst_transferability_uniform.json", "Uniform"),
    ("ETTm2", "results/ettm2_patchtst_transferability_ten.json", "TEN"),
    ("ETTm2", "results/ettm2_patchtst_transferability_correlation.json", "Correlation"),
    ("ETTm2", "results/ettm2_patchtst_transferability_uniform.json", "Uniform"),
]


def load_metrics(path: str) -> dict:
    payload = json.loads(Path(path).read_text())
    return payload["test_metrics"]


def main() -> None:
    header = "| Dataset | Method | MSE | MAE | CRPS | Mean Q | Top-1 |"
    rule = "|---|---|---:|---:|---:|---:|---:|"
    rows = [header, rule]

    for dataset, path, method in RESULT_SPECS:
        metrics = load_metrics(path)
        rows.append(
            "| {dataset} | {method} | {mse:.6f} | {mae:.6f} | {crps:.6f} | {mean_q:.6f} | {top1:.6f} |".format(
                dataset=dataset,
                method=method,
                mse=metrics["forecast_mse"],
                mae=metrics["forecast_mae"],
                crps=metrics["forecast_crps"],
                mean_q=metrics["forecast_mean_quantile_loss"],
                top1=metrics["top1_alignment"],
            )
        )

    print("\n".join(rows))


if __name__ == "__main__":
    main()
