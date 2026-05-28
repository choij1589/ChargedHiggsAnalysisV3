#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../automize/load_masspoints.sh
source "${SCRIPT_DIR}/automize/load_masspoints.sh"

for masspoint in "${MASSPOINTs_PARTICLENET[@]}"; do
    echo "[collectThresholds] ${masspoint}"
    python3 "${SCRIPT_DIR}/python/collectThresholds.py" --masspoint "${masspoint}" "$@"
done
