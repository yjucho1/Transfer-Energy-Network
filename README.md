# Transfer Energy Network

Energy-based transferability learning for probabilistic long-term time series forecasting.

## Problem Definition

This repository defines the research problem as multivariate long-term forecasting (LTSF).

Given an input window of length `L`, the model predicts a future horizon `H` for a target series while leveraging transferable information from heterogeneous source series or source-specific components. The central question is not simply which source looks similar, but which source improves long-horizon forecasting performance on the target domain.

The default protocol in this repository follows common LTSF benchmark settings:

- task: `long_term_forecast`
- feature mode: multivariate-to-multivariate (`M`)
- default look-back: `seq_len = 96`
- standard prediction horizons: `96, 192, 336, 720`

## Dataset Assumption

We assume the experimental data is prepared with ProbTS and stored under `./datasets`.

ProbTS documents long-term forecasting dataset preparation with:

```bash
bash scripts/prepare_datasets.sh "./datasets"
```

and then configures long-term forecasting runs with dataset keys such as `etth1`, `etth2`, `ettm1`, `ettm2`, `traffic_ltsf`, `electricity_ltsf`, `exchange_ltsf`, `illness_ltsf`, and `weather_ltsf`. [Source](https://github.com/microsoft/ProbTS)

The earlier Google Drive link can still be viewed as a compatible benchmark-bundle assumption, but the repository is now organized around the ProbTS preparation path and naming convention.

Under that assumption, the benchmark suite in this repository targets the standard LTSF datasets commonly used together in that ecosystem:

- `etth1`, `etth2`, `ettm1`, `ettm2`
- `electricity_ltsf`
- `traffic_ltsf`
- `weather_ltsf`
- `exchange_ltsf`
- `illness_ltsf`

The dataset registry is encoded in [datasets.py](/Users/lisa.cho/Transfer-Energy-Network/src/transfer_energy_network/data/datasets.py:1), and experiment metadata such as `dataset_bundle`, `dataset_name`, `file_name`, `seq_len`, and `pred_len` is now part of each config.

For dataset descriptions, long-horizon benchmark references such as Nixtla's long-horizon dataset page list the same family of groups: `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `ECL`, `Exchange`, `Traffic`, `Weather`, and `ILI`. [Source](https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon2.html)

## Motivation

Transfer learning can help forecasting models adapt quickly when a target time series has limited history. In practice, however, heterogeneous source series often cause negative transfer, especially in long-horizon settings where mismatched trend, seasonality, and domain dynamics are amplified across longer prediction windows. The usual workaround is to preselect source series with hand-crafted similarity measures such as correlation, dynamic time warping, or latent-distance heuristics. Those signals may correlate with similarity, but they do not directly answer the question that matters most:

Which source component will improve target probabilistic forecasting performance?

This repository frames source selection as a learned energy-based decision problem. A Transfer Energy Network (TEN) assigns:

- low energy to source components that improve target validation likelihood
- high energy to source components that harm adaptation

The resulting energy scores can then be used to:

- select a subset of source adapters
- reweight source gradients during adaptation
- gate transferred representations before the target forecasting head

## Core Idea

Let `x_tgt` denote target context windows and let `s_i` denote a candidate transferable source component. The transferability scorer learns an energy

`E_theta(x_tgt, s_i) -> R`

where lower values imply higher transfer utility. A normalized weight can be obtained from the negative energies:

`w_i = softmax(-E_theta(x_tgt, s_i) / tau)`

These weights are used to aggregate source information before passing it to a probabilistic forecaster.

## Minimal Architecture In This Starter

This starter keeps the implementation intentionally small:

1. A target encoder maps each target window into a compact representation.
2. A source encoder maps candidate source summaries into the same latent space.
3. An energy MLP scores target-source compatibility.
4. A weighting module converts energies into transfer weights.
5. A probabilistic forecasting head predicts a Gaussian mean and scale.

The code is designed so we can later swap in:

- adapter banks instead of simple source embeddings
- gradient-level transfer instead of representation-level transfer
- richer likelihood heads such as Negative Binomial, Student-t, or quantile mixtures

## Validation-Delta Supervision

To learn actual transfer utility rather than static similarity, this starter now includes a small supervision recipe for the energy scorer.

For each target episode and candidate source, define a validation delta:

`Delta_i = val_metric(target with source_i) - val_metric(target alone)`

When the validation metric is a likelihood-like score where larger is better, positive `Delta_i` means the source helped and negative `Delta_i` means it hurt. We transform these deltas into a soft target distribution and train the energy model so that:

- lower energy aligns with larger positive validation deltas
- higher energy aligns with harmful or unhelpful sources

In code, the joint objective is:

- forecast loss: Gaussian negative log-likelihood on the target horizon
- transfer loss: cross-entropy between `softmax(Delta / tau)` and `softmax(-E / tau)`

For the current real CSV loader, `Delta_i` is approximated more directly than simple correlation: each source is converted into a target-scale proxy forecast, compared against a target-only persistence baseline on the future horizon, and then slightly regularized with history and trend agreement scores.

## Repository Layout

```text
configs/
  ten_synthetic.toml
  correlation_synthetic.toml
  uniform_synthetic.toml
scripts/
  run_experiment.py
src/transfer_energy_network/
  __init__.py
  config.py
  data/
    __init__.py
    datasets.py
    episodes.py
    ltsf.py
  models/
    __init__.py
    baselines.py
    energy_model.py
    forecasting.py
    weighting.py
  training/
    __init__.py
    evaluation.py
    experiment.py
    losses.py
    trainer.py
examples/
  smoke_test.py
  train_synthetic.py
pyproject.toml
```

## Paper-Oriented Experiment Design

논문 작성을 위한 실험 구조는 아래 흐름으로 설계되어 있습니다.

1. `configs/*.toml`에서 실험 설정을 고정합니다.
2. `data/datasets.py`와 `config.py`가 long-term forecasting dataset metadata를 고정합니다.
3. `data/episodes.py`가 target-source episodic split을 생성합니다.
This repository now also includes a real CSV-based LTSF loader in [ltsf.py](/Users/lisa.cho/Transfer-Energy-Network/src/transfer_energy_network/data/ltsf.py:1), which reads the ProbTS-prepared files under `./datasets`, builds train/val/test sliding windows, normalizes using training statistics, and converts them into TEN training episodes.
4. `experiment.py`가 TEN 또는 baseline selector를 실행합니다.
5. `evaluation.py`가 forecasting NLL, source top-1 alignment, oracle hit rate를 계산합니다.
6. 각 실험은 `results/*.json`으로 저장되어 표와 ablation 정리에 바로 사용할 수 있습니다.

이 구조로 다음 비교가 쉬워집니다.

- TEN vs correlation vs uniform
- ETTh1/ETTh2/ETTm1/ETTm2/Electricity/Traffic/Weather/Exchange/ILI across the same protocol
- energy temperature ablation
- transfer loss weight ablation
- source pool size 변화
- prediction horizon `96/192/336/720` 변화
- noisy or heterogeneous source 비율 변화

## Next Research Steps

- Replace synthetic source summaries with learned source adapters.
- Compare TEN against correlation, DTW, and representation-distance baselines.
- Extend the forecasting head beyond Gaussian likelihoods for count or intermittent series.
- Add real target adaptation episodes and derive validation deltas from actual fine-tuning.

## Quick Start

If your environment already has PyTorch installed:

```bash
python3 examples/smoke_test.py
```

The example builds a tiny TEN model, computes energy-based source weights, and returns Gaussian forecasting parameters for a synthetic batch.

To exercise the joint training scaffold:

```bash
python3 examples/train_synthetic.py
```

This synthetic example creates episodes with one more-useful source per target batch, uses validation deltas as supervision, and optimizes the forecasting head and energy scorer together.

To run a paper-style experiment and save a summary:

```bash
python3 scripts/run_experiment.py configs/ten_synthetic.toml
python3 scripts/run_experiment.py configs/correlation_synthetic.toml
python3 scripts/run_experiment.py configs/uniform_synthetic.toml
```

Each run writes a JSON summary with train history, validation history, and test metrics.

When you replace the current synthetic episode generator with real loaders, the expected dataset location is:

```text
./datasets/
  ETT-small/ETTh1.csv
  ETT-small/ETTh2.csv
  ETT-small/ETTm1.csv
  ETT-small/ETTm2.csv
  electricity/electricity.csv
  traffic/traffic.csv
  weather/weather.csv
  exchange_rate/exchange_rate.csv
  illness/national_illness.csv
```

This path layout matches the files prepared in the current workspace by the ProbTS dataset script.
