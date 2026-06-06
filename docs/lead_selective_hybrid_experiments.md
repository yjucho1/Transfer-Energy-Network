# Lead-Selective Hybrid Experiments

## Overview

This document summarizes the current `lead_selective` model family, the repeatable experiment runners added around it, and the main ETT benchmark results obtained so far.

The current direction is:

- build a `source-aware` forecasting branch with a lead-aware source proposer
- keep a strong frozen `target-only` base forecaster
- use a selective gate to control how much the final prediction should rely on the source-aware branch

This makes the model a selective forecasting system rather than a pure source selector.

## Model

The model is implemented in [lead_selective_hybrid.py](/Users/lisa.cho/Transfer-Energy-Network/src/transfer_energy_network/models/lead_selective_hybrid.py).

### Components

1. `Frozen target-only forecaster`
- a pretrained PatchTST Gaussian forecaster
- used as the conservative fallback path

2. `Lead-aware source proposer`
- target-conditioned cross-attention over source candidate histories
- learned relative lag bias to model lead-lag structure
- outputs source scores and a soft source weight distribution

3. `Source-aware forecaster`
- builds a weighted pooled source sequence from proposer weights
- predicts the same forecasting target using target history plus pooled source sequence

4. `Selective gate`
- softly interpolates between target-only and source-aware forecasts
- final prediction:
  - `final = (1 - gate) * target_only + gate * source_aware`

### Gate feature modes

Two gate feature variants were tested.

`basic`
- proposer entropy
- proposer top-1/top-2 gap
- norm of forecast difference between source-aware and target-only outputs

`attn`
- all `basic` features
- plus attention entropy from the proposer cross-attention map

## Training Setup

### Shared forecasting setup

- datasets: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`
- target setting: univariate target prediction
- source pool: same-dataset other variables plus variables from external ETT datasets
- context length: `96`
- prediction lengths tested:
  - `96`
  - `192`
  - `336`
  - `720`

### Optimization

- target-only base checkpoint is loaded and frozen
- proposer, source-aware forecaster, and gate are trained end-to-end
- training loss:
  - forecasting loss on final blended prediction
  - plus gate usage penalty
- final stable setting:
  - `gate_aux_weight = 0.0`

### Compact sweep space

The repeatable sweep currently uses:

- proposer temperature: `{0.7, 1.0}`
- gate usage penalty: `{0.001, 0.005}`
- gate feature mode: `{basic, attn}`
- gate auxiliary weight: `{0.0}`

## Runners

### Main sweep runner

[run_lead_selective_hybrid_ett_experiments.py](/Users/lisa.cho/Transfer-Energy-Network/scripts/run_lead_selective_hybrid_ett_experiments.py)

Supports:

- dataset sweeps
- prediction-length sweeps
- compact hyperparameter sweeps
- JSON and Markdown summary output

Example:

```bash
python3 scripts/run_lead_selective_hybrid_ett_experiments.py \
  --datasets etth1 etth2 ettm1 ettm2 \
  --pred-lens 96 192 336 720 \
  --epochs 8 \
  --early-stopping-patience 3 \
  --proposer-temperature-sweep 0.7 1.0 \
  --gate-usage-penalty-sweep 0.001 0.005 \
  --gate-aux-weight-sweep 0.0 \
  --gate-feature-mode-sweep basic attn
```

### Fixed best-config runner

[run_best_lead_selective_configs.py](/Users/lisa.cho/Transfer-Energy-Network/scripts/run_best_lead_selective_configs.py)

This runner replays fixed per-dataset best settings from TOML files:

- [lead_selective_hybrid_etth1.toml](/Users/lisa.cho/Transfer-Energy-Network/configs/lead_selective_hybrid_etth1.toml)
- [lead_selective_hybrid_etth2.toml](/Users/lisa.cho/Transfer-Energy-Network/configs/lead_selective_hybrid_etth2.toml)
- [lead_selective_hybrid_ettm1.toml](/Users/lisa.cho/Transfer-Energy-Network/configs/lead_selective_hybrid_ettm1.toml)
- [lead_selective_hybrid_ettm2.toml](/Users/lisa.cho/Transfer-Energy-Network/configs/lead_selective_hybrid_ettm2.toml)

## Main Results

### Best compact-sweep result at `pred=96`

Stored in:

- [lead_selective_hybrid_ett_sweep_summary.json](/Users/lisa.cho/Transfer-Energy-Network/results/lead_selective_hybrid_ett_sweep_summary.json)
- [lead_selective_hybrid_ett_sweep_summary.md](/Users/lisa.cho/Transfer-Energy-Network/results/lead_selective_hybrid_ett_sweep_summary.md)

Best configurations:

- `ETTh1`: `attn`, `temp=1.0`, `usage=0.005`
- `ETTh2`: `basic`, `temp=0.7`, `usage=0.005`
- `ETTm1`: `attn`, `temp=1.0`, `usage=0.001`
- `ETTm2`: `basic`, `temp=1.0`, `usage=0.005`

All four datasets improved over target-only on `MSE`, `MAE`, `CRPS`, and mean quantile loss under this compact sweep.

### Multi-horizon replay using dataset-specific best settings

Stored in:

- [lead_selective_hybrid_best_configs_multi_pred.json](/Users/lisa.cho/Transfer-Energy-Network/results/lead_selective_hybrid_best_configs_multi_pred.json)

Important note:

- for `pred=192/336/720`, we replayed the dataset-specific best settings found at `pred=96`
- this is not a full horizon-specific sweep for every horizon

#### Summary by dataset

`ETTh1`
- improved at `96` and `720`
- degraded at `192` and `336`

`ETTh2`
- improved at `96`, `192`, `336`
- degraded at `720`

`ETTm1`
- improved at all tested horizons
- especially strong gain at `720`

`ETTm2`
- improved at all tested horizons

### Horizon-specific follow-up for degraded cases

Follow-up sweeps were run for:

- `ETTh1 @ 192`
- `ETTh1 @ 336`
- `ETTh2 @ 720`

Stored in:

- [lead_selective_hybrid_etth1_pred192_hsweep.json](/Users/lisa.cho/Transfer-Energy-Network/results/lead_selective_hybrid_etth1_pred192_hsweep.json)
- [lead_selective_hybrid_etth1_pred336_hsweep.json](/Users/lisa.cho/Transfer-Energy-Network/results/lead_selective_hybrid_etth1_pred336_hsweep.json)
- [lead_selective_hybrid_etth2_pred720_hsweep.json](/Users/lisa.cho/Transfer-Energy-Network/results/lead_selective_hybrid_etth2_pred720_hsweep.json)

Takeaway:

- `ETTh1 @ 192` remained degraded after horizon-specific tuning
- `ETTh1 @ 336` remained degraded and could become slightly worse
- `ETTh2 @ 720` partially recovered but still stayed below target-only

This suggests that some failures are not only due to reusing `pred=96` settings, but also reflect horizon-dependent limits of the current source-aware branch and soft blending policy.

## Interpretation

### What currently works

- the selective hybrid works especially well on `ETTm1` and `ETTm2`
- `ETTm2` is the most consistently source-friendly dataset
- `ETTm1 @ 720` shows the largest gain among tested replay settings

### What currently fails

- some horizons on `ETTh1` and `ETTh2` remain harmful even after retuning
- this points to either:
  - weak horizon-specific transferability
  - source-aware branch mismatch at medium/long horizons
  - soft blending being too permissive when source quality is ambiguous

## Current Practical Recommendation

If reproducing current best behavior:

- use the dataset-specific best configs at `pred=96`
- treat `ETTm1` and `ETTm2` as the strongest positive cases
- be cautious on:
  - `ETTh1 @ 192`
  - `ETTh1 @ 336`
  - `ETTh2 @ 720`

## Next Suggested Experiments

1. Hard reject-to-base gating instead of only soft blending
2. Source-aware branch standalone evaluation on degraded horizons
3. Full horizon-specific best-config search for all `96/192/336/720`
4. Separate analysis of gate activation statistics by horizon and dataset
