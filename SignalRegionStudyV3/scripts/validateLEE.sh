#!/bin/bash
#
# Run Step 5 LEE validation.

set -euo pipefail

MASSPOINT="MHc70_MA18"
REFERENCE_MASSPOINT="MHc160_MA50"
START_TOY=1
NTOYS=1000
DEBUG=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --masspoint)
            MASSPOINT="$2"
            shift 2
            ;;
        --reference-masspoint)
            REFERENCE_MASSPOINT="$2"
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
            echo "Usage: $0 [--masspoint MASSPOINT] [--reference-masspoint MASSPOINT] [--start-toy N] [--ntoys N] [--debug]"
            echo "  --masspoint MASSPOINT            LEE mass point [default: MHc70_MA18]"
            echo "  --reference-masspoint MASSPOINT  Background sample comparison point [default: MHc160_MA50]"
            echo "  --start-toy N                    First toy index [default: 1]"
            echo "  --ntoys N                        Number of toys to validate [default: 1000]"
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

ARGS=(
    --masspoint "$MASSPOINT"
    --reference-masspoint "$REFERENCE_MASSPOINT"
    --start-toy "$START_TOY"
    --ntoys "$NTOYS"
)
if [[ "$DEBUG" == true ]]; then
    ARGS+=(--debug)
fi

echo "============================================================"
echo "LEE Step 5: validate toy chain"
echo "Mass point: $MASSPOINT"
echo "Reference mass point: $REFERENCE_MASSPOINT"
echo "Toy range: $START_TOY-$((START_TOY + NTOYS - 1))"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

python3 python/validateLEE.py "${ARGS[@]}"

echo "============================================================"
echo "LEE Step 5 completed: results/lee/validation"
echo "Time: $(date)"
echo "============================================================"
