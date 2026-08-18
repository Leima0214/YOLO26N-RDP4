#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
CONFIG="configs/experiments/japan4_cleanv3_schm_100e.yaml"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs

command -v nvidia-smi >/dev/null
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
test -x "${PYTHON_BIN}"
test -f "${CONFIG}"
test -f "configs/japan4_clean_v3_remote.yaml"
test -d "/Japan4-V3/Japan4-cleanV3/images/train"
test -d "/Japan4-V3/Japan4-cleanV3/images/val"

echo "GIT_COMMIT=$(git rev-parse HEAD)"
echo "TEST_SEALED=val-only"
sed -n '1,240p' "${CONFIG}"

"${PYTHON_BIN}" -u scripts/train_schm_family.py \
  --variant schm \
  --config "${CONFIG}" \
  2>&1 | tee "logs/schm_cleanv3_100e_${STAMP}.log"
