#!/bin/bash
#
# Generate Step 2 LEE nominal-background toy TTrees.

set -euo pipefail

MASSPOINT="MHc70_MA18"
ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --masspoint)
            MASSPOINT="$2"
            shift 2
            ;;
        --toy)
            ARGS+=(--toy "$2")
            shift 2
            ;;
        --ntoys)
            ARGS+=(--ntoys "$2")
            shift 2
            ;;
        --start-toy)
            ARGS+=(--start-toy "$2")
            shift 2
            ;;
        --force)
            ARGS+=(--force)
            shift
            ;;
        --debug)
            ARGS+=(--debug)
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--masspoint MASSPOINT] [--toy N | --start-toy N --ntoys N] [--force] [--debug]"
            echo "  --masspoint MASSPOINT  LEE mass point [default: MHc70_MA18]"
            echo "  --toy N                Generate one toy index"
            echo "  --start-toy N          First toy index for --ntoys [default: 1]"
            echo "  --ntoys N              Number of toys to generate [default: 1]"
            echo "  --force                Overwrite complete existing toys"
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
echo "LEE Step 2: generate nominal background toys"
echo "Mass point: $MASSPOINT"
echo "Args: ${ARGS[*]}"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

python3 python/generateLEEToys.py --masspoint "$MASSPOINT" "${ARGS[@]}"

echo "============================================================"
echo "LEE Step 2 completed: LEE/${MASSPOINT}/toys"
echo "Time: $(date)"
echo "============================================================"
