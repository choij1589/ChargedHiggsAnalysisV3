#!/bin/bash
#
# Collect Step 4 LEE toy fits and compute the global p-value.

set -euo pipefail

MASSPOINT="MHc70_MA18"
START_TOY=1
NTOYS=1000
DEBUG=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --masspoint)
            MASSPOINT="$2"
            shift 2
            ;;
        --start-toy)
            START_TOY="$2"
            shift 2
            ;;
        --ntoys)
            NTOYS="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--masspoint MASSPOINT] [--start-toy N] [--ntoys N] [--debug]"
            echo "  --masspoint MASSPOINT  Observed maximum-excess mass point [default: MHc70_MA18]"
            echo "  --start-toy N          First toy index [default: 1]"
            echo "  --ntoys N              Number of toy fit outputs to collect [default: 1000]"
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

ARGS=(--masspoint "$MASSPOINT" --start-toy "$START_TOY" --ntoys "$NTOYS")
if [[ "$DEBUG" == true ]]; then
    ARGS+=(--debug)
fi

echo "============================================================"
echo "LEE Step 4: collect global p-value"
echo "Mass point: $MASSPOINT"
echo "Toy range: $START_TOY-$((START_TOY + NTOYS - 1))"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

python3 python/collectLEE.py "${ARGS[@]}"

echo "============================================================"
echo "LEE Step 4 completed: results/lee"
echo "Time: $(date)"
echo "============================================================"
