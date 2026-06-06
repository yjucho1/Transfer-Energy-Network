#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python3 scripts/run_best_lead_selective_configs.py \
  configs/lead_selective_attended_etth1.toml \
  configs/lead_selective_attended_etth2.toml \
  configs/lead_selective_attended_ettm1.toml \
  configs/lead_selective_attended_ettm2.toml \
  --pred-lens 96 192 336 720 \
  --output results/lead_selective_attended_best_configs_multi_pred.json
