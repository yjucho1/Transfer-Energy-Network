| Dataset | Target-Only MSE | Best Energy-Mandatory Config | Best Energy-Mandatory MSE | MSE Delta | MAE Delta | CRPS Delta | Mean Q Delta | Notes |
|---|---:|---|---:|---:|---:|---:|---:|---|
| `ETTh1` | `0.5362316` | `corr_temp=4.5`, `hybrid_gate_entropy`, `lam=0.2`, `th=0.14` | `0.5338346` | `-0.0023970` | `-0.0006787` | `-0.0005736` | `-0.0002258` | Strictly better than target-only on all four metrics. |
| `ETTh2` | `0.2267056` | `corr_temp=0.7`, `energy_gate_entropy`, `th=3.2` | `0.2266210` | `-0.0000846` | `-0.0000226` | `-0.0000188` | `-0.0000091` | Improvement exists but is very small. |
| `ETTm1` | `0.4034646` | No all-metric win found in current search | `0.4099532` | `+0.0064886` | `+0.0088952` | `+0.0010195` | `+0.0012421` | Current energy-mandatory search did not beat target-only. |
| `ETTm2` | `0.2372661` | `corr_temp=0.7`, `hybrid_gate_entropy`, `lam=0.05`, `th=0.06` | `0.2241268` | `-0.0131393` | `-0.0101215` | `-0.0084380` | `-0.0040097` | Strongest gain among the four datasets. |

## Source

- Raw experiment summary: [energy_mandatory_hybrid_ett_summary.json](/Users/lisa.cho/Transfer-Energy-Network/results/energy_mandatory_hybrid_ett_summary.json)
- Runner: [run_energy_mandatory_hybrid_experiments.py](/Users/lisa.cho/Transfer-Energy-Network/scripts/run_energy_mandatory_hybrid_experiments.py)

## Interpretation

- The most reliable use of the energy model was not direct source ranking, but source-usage control.
- The strongest pattern was: build a source-aware forecast with correlation, then let energy entropy decide whether that forecast should be trusted.
- This worked on `ETTh1`, `ETTh2`, and `ETTm2`, but not yet on `ETTm1`.
