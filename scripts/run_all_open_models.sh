#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

python3 evaluate.py \
  --config configs/dataset_emoset2k.yaml \
  --models configs/models.all.yaml \
  --reasoning-config configs/reasoning.default.yaml \
  --only-source open_source
