#!/bin/bash
#
# runCombinedAsymptotic.sh - Run asymptotic limits for Full Run2 Combined channel
#
# Usage:
#   ./scripts/runCombinedAsymptotic.sh --masspoint MHc130_MA90 --method Baseline --binning uniform
#   ./scripts/runCombinedAsymptotic.sh --masspoint MHc130_MA90 --method ParticleNet --binning extended --partial-unblind
#

set -e

# Default values
MASSPOINT=""
METHOD="Baseline"
BINNING="uniform"
PARTIAL_UNBLIND=false
ERAS=(2016preVFP 2016postVFP 2017 2018)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --masspoint)
            MASSPOINT="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --binning)
            BINNING="$2"
            shift 2
            ;;
        --partial-unblind)
            PARTIAL_UNBLIND=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --masspoint MASSPOINT [--method METHOD] [--binning BINNING] [--partial-unblind]"
            echo ""
            echo "Required:"
            echo "  --masspoint  Signal mass point (e.g., MHc130_MA90)"
            echo ""
            echo "Options:"
            echo "  --method     Template method (Baseline, ParticleNet) [default: Baseline]"
            echo "  --binning    Binning scheme (uniform, extended) [default: uniform]"
            echo "  --partial-unblind  Use partial-unblind templates (score < 0.3)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$MASSPOINT" ]]; then
    echo "ERROR: --masspoint is required"
    exit 1
fi

# Build partial-unblind flag for downstream commands
PARTIAL_UNBLIND_FLAG=""
if [[ "$PARTIAL_UNBLIND" == true ]]; then
    PARTIAL_UNBLIND_FLAG="--partial-unblind"
fi

# Construct binning suffix for display
BINNING_SUFFIX="${BINNING}"
if [[ "$PARTIAL_UNBLIND" == true ]]; then
    BINNING_SUFFIX="${BINNING}_partial_unblind"
fi

echo "Running Combined Asymptotic for ${MASSPOINT} (${METHOD}/${BINNING_SUFFIX})"
echo ""

# Step 1: Combine channels (SR1E2Mu + SR3Mu → Combined) for each era
echo "Step 1: Combining channels for each era..."
for ERA in "${ERAS[@]}"; do
    echo "  Combining channels for ${ERA}..."
    combineDatacards.py --mode channel --era "$ERA" --masspoint "$MASSPOINT" --method "$METHOD" --binning "$BINNING" $PARTIAL_UNBLIND_FLAG
done

# Step 2: Combine eras (4 eras → Run2)
echo ""
echo "Step 2: Combining eras into Run2..."
combineDatacards.py --mode era --channel Combined --masspoint "$MASSPOINT" --method "$METHOD" --binning "$BINNING" --output-era Run2 $PARTIAL_UNBLIND_FLAG

# Step 3: Asymptotic limits
echo ""
echo "Step 3: Running AsymptoticLimits..."
./scripts/runAsymptotic.sh --era Run2 --channel Combined --masspoint "$MASSPOINT" --method "$METHOD" --binning "$BINNING" $PARTIAL_UNBLIND_FLAG

echo ""
echo "Done."
