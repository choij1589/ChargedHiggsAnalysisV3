#!/bin/bash
#
# runCombinedAsymptotic.sh - Run asymptotic limits for Full Run2 Combined channel
#
# Usage:
#   ./scripts/runCombinedAsymptotic.sh MHc130_MA90 Baseline uniform
#

set -e

MP=$1
METHOD=${2:-Baseline}
BINNING=${3:-uniform}
ERAS=(2016preVFP 2016postVFP 2017 2018)

# Get script directory
BASE_DIR="$(dirname "$SCRIPT_DIR")"

# Validate arguments
if [[ -z "$MP" ]]; then
    echo "Usage: $0 MASSPOINT [METHOD] [BINNING]"
    echo "  MASSPOINT: e.g., MHc130_MA90"
    echo "  METHOD: Baseline (default) or ParticleNet"
    echo "  BINNING: uniform (default) or sigma"
    exit 1
fi

echo "Running Combined Asymptotic for ${MP} (${METHOD}/${BINNING})"
echo ""

# Step 1: Combine channels (SR1E2Mu + SR3Mu → Combined) for each era
echo "Step 1: Combining channels for each era..."
for ERA in "${ERAS[@]}"; do
    echo "  Combining channels for ${ERA}..."
    combineDatacards.py --mode channel --era "$ERA" --masspoint "$MP" --method "$METHOD" --binning "$BINNING"
done

# Step 2: Combine eras (4 eras → Run2)
echo ""
echo "Step 2: Combining eras into Run2..."
combineDatacards.py --mode era --channel Combined --masspoint "$MP" --method "$METHOD" --binning "$BINNING" --output-era Run2

# Step 3: Asymptotic limits
echo ""
echo "Step 3: Running AsymptoticLimits..."
./scripts/runAsymptotic.sh --era Run2 --channel Combined --masspoint "$MP" --method "$METHOD" --binning "$BINNING"

echo ""
echo "Done."
