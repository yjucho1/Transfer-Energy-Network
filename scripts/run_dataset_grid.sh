#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <dataset> [seed1 seed2 ...]" >&2
  echo "dataset: etth1 | etth2 | ettm1 | ettm2" >&2
  exit 1
fi

DATASET="$1"
shift
if [[ $# -gt 0 ]]; then
  SEEDS=("$@")
else
  SEEDS=(21 22 23 24 25)
fi
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT

if [[ -f "/Users/lisa.cho/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source /Users/lisa.cho/miniconda3/etc/profile.d/conda.sh
  conda activate lisa_env
fi

cd "${ROOT_DIR}"

for horizon in 96 192 336 720; do
  for seed in "${SEEDS[@]}"; do
    for method in ten correlation uniform; do
      base_config="${CONFIG_DIR}/${method}_${DATASET}_patchtst.toml"
      if [[ ! -f "${base_config}" ]]; then
        echo "missing config: ${base_config}" >&2
        exit 1
      fi

      temp_config="${TMP_DIR}/${method}_${DATASET}_h${horizon}_s${seed}.toml"
      cp "${base_config}" "${temp_config}"
      perl -0pi -e "s/name = \"${DATASET}_patchtst_transferability\"/name = \"${DATASET}_h${horizon}_s${seed}_patchtst_transferability\"/" "${temp_config}"
      perl -0pi -e "s/seed = \\d+/seed = ${seed}/" "${temp_config}"
      perl -0pi -e "s/pred_len = \\d+/pred_len = ${horizon}/" "${temp_config}"

      echo "RUNNING dataset=${DATASET} horizon=${horizon} seed=${seed} method=${method}"
      python "${ROOT_DIR}/scripts/run_experiment.py" "${temp_config}"
    done
  done
done
