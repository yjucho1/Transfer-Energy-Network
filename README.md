# Transfer Energy Network

Energy-based transferability learning for probabilistic long-term time series forecasting.

## Overview

This repository studies when source time series help or hurt target forecasting on the ETT family.

The current codebase focuses on:

- probabilistic long-term forecasting with a PatchTST backbone
- independent per-variable forecasting episodes
- source-conditioned residual correction on top of a frozen target-only base model
- structural energy learning over forecast corrections

The current question is:

How should source information be used to improve a target-only forecast, and can an energy model learn to prefer better forecast corrections?

## Current Problem Setup

The active ETTh1 experiments use:

- lookback: `seq_len = 96`
- horizon: `pred_len = 96`
- feature mode: multivariate CSV input, but independent univariate forecasting episodes

Each training sample is:

- one target variable history of length `96`
- one target future of length `96`
- a source pool containing:
  - the other variables from the same dataset
  - variables from external ETT datasets

For ETTh1, the current source pool size is `27`:

- `6` same-dataset variables
- `21` cross-dataset variables from `ETTh2`, `ETTm1`, and `ETTm2`

This means the current default setting is:

- target context: univariate
- prediction target: univariate
- source pool: same-dataset others + other-dataset variables

## Current Model

The current TEN model is a residual-correction model.

### Base Forecast

The base forecaster is a target-only PatchTST model:

\[
\hat y_{\text{base}} = F_{\text{base}}(t)
\]

In the current implementation, this base model is:

- first trained as `target_only`
- loaded into TEN from the saved best checkpoint
- frozen during TEN training

So the TEN base forecast is exactly the same model as the target-only baseline.

### Source Residual

TEN then predicts a source-conditioned correction:

\[
\Delta \hat y = F_{\text{res}}(t, S)
\]

with a learned scalar gate:

\[
\alpha = \sigma(a)
\]

and the final forecast is:

\[
\hat y = \hat y_{\text{base}} + \alpha \Delta \hat y
\]

### Current Source Aggregation

The current ETTh1 TEN config uses:

- `source_pooling = "average"`

So sources are encoded independently and then averaged:

\[
z_{s_i} = \mathrm{Enc}_s(s_i), \qquad
z_S = \frac{1}{K}\sum_{i=1}^K z_{s_i}
\]

For the residual forecasting branch, the raw source sequences are also averaged before being passed to the residual PatchTST branch.

## Two Energy Formulations Explored

This repository has explored two closely related but importantly different energy objectives.

### 1. Plausibility-Oriented Energy

The first formulation treats energy as a plausibility score:

\[
E_\theta(t,S,Y)
\]

or, in the residual version,

\[
E_{\text{corr}}(t,S,\Delta y)
\]

The key idea is:

- lower energy for more plausible futures or corrections
- higher energy for less plausible ones

This is the SEAL-style view of energy as a trainable structural loss.

In earlier versions of the project, this was implemented by contrasting:

- ground-truth future or correction
- current predicted future or correction

This formulation successfully learned energy separation, but in ETTh1 the resulting plausibility improvement did not reliably translate into better forecasting metrics.

### 2. Metric-Guided Energy

The second formulation tries to align the energy model more directly with forecasting quality.

Instead of defining positives and negatives only by structural plausibility, it defines them using actual forecast metric quality over a candidate correction set.

For a candidate correction `\Delta y`, the model evaluates the quality of:

\[
\hat y(\Delta y) = \hat y_{\text{base}} + \Delta y
\]

and then chooses the best candidate according to the forecast metric.

In the current code, positives and negatives are selected from:

- the current predicted correction
- Monte Carlo sampled corrections from the predictive distribution

The best candidate under a sample-wise forecasting metric becomes the positive example, and the others are treated as negatives in an NCE loss.

This gives a more metric-aligned formulation:

- plausibility-oriented energy asks: “is this correction structurally reasonable?”
- metric-guided energy asks: “does this correction actually improve the forecast metric?”

## Structural Correction Energy

The current energy model scores forecast corrections rather than full futures:

\[
E_{\text{corr}}(t, S, \Delta y)
\]

where:

- `t` is the target past
- `S` is the pooled source context
- `\Delta y` is a candidate correction to the frozen base forecast

The ground-truth correction is:

\[
\Delta y^* = y - \hat y_{\text{base}}
\]

## Current Training Objective

### Loss-Net

The current loss-net uses the metric-guided energy version with an NCE objective.

For each batch item, the code builds a candidate correction set:

- the current predicted correction
- several sampled corrections from the predictive distribution

The candidate with the best sample-wise forecast metric is used as the positive example, and the rest are negatives.

In the current implementation, this positive/negative selection is metric-based and uses sample-wise CRPS-style scoring.

The NCE loss is:

\[
\mathcal{L}_{E}^{\text{NCE}}
=
-\log
\frac{
\exp(-E_{\text{corr}}(t,S,\Delta y^+)/\tau)
}{
\sum_{\Delta y \in \mathcal{C}} \exp(-E_{\text{corr}}(t,S,\Delta y)/\tau)
}
\]

where `\Delta y^+` is the metric-best candidate correction from the candidate set `\mathcal{C}`.

### Task-Net

The task objective is:

\[
\mathcal{L}_{\text{task}}
=
\lambda_f \,\ell(y,\hat y)
+
\lambda_b \,\ell(y,\hat y_{\text{base}})
+
\lambda_E \,\mathbb{E}_{\Delta Y \sim p_\phi}[E_{\text{corr}}(t,S,\Delta Y)]
\]

In practice:

- `\ell` is `CRPS`
- the expectation term is approximated with Monte Carlo samples from the predictive distribution
- the sampled correction energies are weighted by metric quality

### Alternating Optimization

Each training step alternates:

1. loss-net update

- updates: `future_encoder`, `trajectory_energy_mlp`
- freezes: target encoder, source encoder, residual forecaster, base forecaster

2. task-net update

- updates: target encoder, source encoder, residual forecaster
- base forecaster remains frozen
- loss-net remains frozen

## Current ETTh1 Status

The current ETTh1 comparison is fully fair:

- the same target-only checkpoint is used as the frozen base for TEN, correlation, and uniform
- all methods operate on the same independent all-variables setup

### Latest ETTh1 Results

The table below corresponds to the current **metric-guided energy** setup:

- frozen target-only base forecaster
- source residual correction
- correction energy trained with metric-guided NCE
- task-side Monte Carlo energy regularization
- energy-based source pooling for TEN
- fair frozen-base comparison against target-only, correlation, and uniform

| Method | MSE | MAE | CRPS | Mean Q |
|---|---:|---:|---:|---:|
| Correlation | 0.53508 | 0.49932 | 0.37428 | 0.17009 |
| Target-only | 0.53623 | 0.49920 | 0.37408 | 0.16993 |
| Uniform | 0.53712 | 0.50045 | 0.37497 | 0.17037 |
| TEN | 0.53895 | 0.50157 | 0.37570 | 0.17067 |

### ETTh1: Two Energy Formulations

For ETTh1, we also keep track of the two TEN energy variants directly:

| TEN Energy Variant | MSE | MAE | CRPS | Mean Q |
|---|---:|---:|---:|---:|
| Plausibility-oriented energy | 0.53823 | 0.50120 | 0.37549 | 0.17059 |
| Metric-guided energy | 0.53895 | 0.50157 | 0.37570 | 0.17067 |

Here:

- **plausibility-oriented energy** means the energy model is trained to assign lower energy to structurally plausible corrections, using the earlier GT-vs-predicted correction contrastive formulation
- **metric-guided energy** means positive and negative corrections are chosen using forecast-metric quality over the candidate correction set

In the current ETTh1 setting, the two formulations are extremely close, which supports the broader conclusion that ETTh1 offers only limited headroom for source-conditioned correction.

### Current Interpretation

For ETTh1 in the current independent setting:

- all methods are very close
- correlation is slightly best
- TEN does not outperform target-only or correlation
- the source residual gate stays small

This table should be read as the result of the **metric-guided energy formulation**, not the earlier plausibility-only formulation.

The current working conclusion is:

In ETTh1 all-variables independent forecasting, source-conditioned correction has very limited headroom, so increasingly sophisticated energy objectives do not translate into clear forecasting gains.

More concretely:

- the energy network does learn to distinguish residual forecasting corrections under both the plausibility-oriented and metric-guided formulations
- however, improved correction-energy scores do not translate into better final forecasting metrics in ETTh1, and the learned source-conditioned correction branch is typically neutral or slightly noisy relative to simpler baselines

## Diagnostics

The most useful current diagnostics are structural-energy diagnostics rather than source-ranking diagnostics.

### Delta-E

We use:

\[
\Delta E_\theta(t,S,Y)
=
E_\theta(t,S,Y)-E_\theta(t,\varnothing,Y)
\]

This measures how much the source context changes the energy of a candidate correction or future.

Current ETTh1 diagnosis shows:

- source context consistently lowers correction energy
- but that lower energy does not correlate strongly with sample-wise forecast gain

So the main bottleneck is not “source is useless,” but rather:

energy-space plausibility improvement does not cleanly translate into forecasting-metric improvement.

This is the clearest difference between the two energy formulations:

- plausibility-oriented energy can show that source information makes corrections look more reasonable
- metric-guided energy asks whether those corrections actually improve CRPS or MSE
- in practice on ETTh1, both formulations learn a correction discriminator, but neither turns that signal into a meaningful forecasting gain

In the current ETTh1 setting, both views suggest that source usefulness exists, but its headroom is small and difficult for TEN to exploit better than simple correlation-based weighting.

## Repository Layout

```text
configs/
  ten_etth1_patchtst.toml
  ten_etth2_patchtst.toml
  ten_ettm1_patchtst.toml
  ten_ettm2_patchtst.toml
  target_only_*.toml
  correlation_*.toml
  uniform_*.toml
scripts/
  run_experiment.py
  plot_sample_forecast.py
  diagnose_structural_energy.py
  diagnose_source_contribution.py
  plot_structural_energy_diagnosis.py
src/transfer_energy_network/
  config.py
  data/
    datasets.py
    episodes.py
    ltsf.py
  models/
    baselines.py
    energy_model.py
    forecasting.py
    target_only.py
    weighting.py
  training/
    evaluation.py
    experiment.py
    losses.py
    trainer.py
results/
```

## Environment

```bash
source /Users/lisa.cho/miniconda3/etc/profile.d/conda.sh
conda activate lisa_env
cd /Users/lisa.cho/Transfer-Energy-Network
```

## Main Commands

Run ETTh1 TEN:

```bash
python scripts/run_experiment.py configs/ten_etth1_patchtst.toml
```

Run ETTh1 target-only:

```bash
python scripts/run_experiment.py configs/target_only_etth1_patchtst.toml
```

Run ETTh1 correlation:

```bash
python scripts/run_experiment.py configs/correlation_etth1_patchtst.toml
```

Run ETTh1 uniform:

```bash
python scripts/run_experiment.py configs/uniform_etth1_patchtst.toml
```

Run structural energy diagnosis:

```bash
python scripts/diagnose_structural_energy.py \
  configs/ten_etth1_patchtst.toml \
  --output results/etth1_structural_energy_diagnosis.json
```

Run source contribution diagnosis:

```bash
python scripts/diagnose_source_contribution.py \
  configs/ten_etth1_patchtst.toml \
  --output results/etth1_source_contribution_diagnosis.json
```

## Notes

- `label_len` remains in config for compatibility but is not used by the active PatchTST path.
- The older `validation_delta`-based ranking pipeline has been removed from the current code path.
- ETTh1 is now an independent all-variables benchmark rather than the earlier `OT`-only setup.
