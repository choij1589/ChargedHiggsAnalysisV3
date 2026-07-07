#!/bin/bash
#
# Build the Step 1 LEE background generation model.

set -euo pipefail

MASSPOINT="MHc70_MA18"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --masspoint)
            MASSPOINT="$2"
            shift 2
            ;;
        --debug)
            EXTRA_ARGS+=(--debug)
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--masspoint MASSPOINT] [--debug]"
            echo "  --masspoint MASSPOINT  LEE generation-model mass point [default: MHc70_MA18]"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"
export PYTHONPATH="${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source ./setup.sh

echo "============================================================"
echo "LEE Step 1: prepare background generation model"
echo "Mass point: $MASSPOINT"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

python3 python/prepareLEEModel.py --masspoint "$MASSPOINT" "${EXTRA_ARGS[@]}"

echo "============================================================"
echo "LEE Step 1 completed: LEE/${MASSPOINT}/model"
echo "Time: $(date)"
echo "============================================================"
