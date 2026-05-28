#!/usr/bin/env bash

set -euo pipefail

"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_dataset_grid.sh" etth2 "$@"
