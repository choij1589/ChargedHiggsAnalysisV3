#!/bin/bash
#
# Project and fit one Step 3 LEE toy across the configured trial set.

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
        --force)
            ARGS+=(--force)
            shift
            ;;
        --keep-workdir)
            ARGS+=(--keep-workdir)
            shift
            ;;
        --debug)
            ARGS+=(--debug)
            shift
            ;;
        --help|-h)
            echo "Usage: $0 --toy N [--masspoint MASSPOINT] [--force] [--keep-workdir] [--debug]"
            echo "  --toy N                Toy index to project and fit"
            echo "  --masspoint MASSPOINT  LEE generation-model mass point [default: MHc70_MA18]"
            echo "  --force                Overwrite complete existing fit output"
            echo "  --keep-workdir         Keep staged per-trial fit directories"
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
echo "LEE Step 3: project toy and fit trial scan"
echo "Mass point: $MASSPOINT"
echo "Args: ${ARGS[*]}"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

python3 python/fitLEEToy.py --masspoint "$MASSPOINT" "${ARGS[@]}"

echo "============================================================"
echo "LEE Step 3 completed: LEE/${MASSPOINT}/fits"
echo "Time: $(date)"
echo "============================================================"
