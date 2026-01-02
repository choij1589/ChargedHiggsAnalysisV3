#!/bin/bash
#
# runImpacts.sh - Calculate systematic uncertainty impacts
#
# Usage:
#   ./runImpacts.sh --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
#   ./runImpacts.sh --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform --condor
#

set -e

# Default values
ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
BINNING="uniform"
PARTIAL_UNBLIND=false
CONDOR=false
PARALLEL=16
DRY_RUN=false
VERBOSE=false
DO_INITIAL=true
DO_FITS=true
DO_PLOT=true

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --era)
            ERA="$2"
            shift 2
            ;;
        --channel)
            CHANNEL="$2"
            shift 2
            ;;
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
        --condor)
            CONDOR=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --skip-initial)
            DO_INITIAL=false
            shift
            ;;
        --skip-fits)
            DO_FITS=false
            shift
            ;;
        --plot-only)
            DO_INITIAL=false
            DO_FITS=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --era        Data-taking period"
            echo "  --channel    Analysis channel (SR1E2Mu, SR3Mu, Combined)"
            echo "  --masspoint  Signal mass point (e.g., MHc130_MA90)"
            echo ""
            echo "Options:"
            echo "  --method     Template method (Baseline, ParticleNet) [default: Baseline]"
            echo "  --binning    Binning scheme (uniform, extended) [default: uniform]"
            echo "  --partial-unblind  Use partial-unblind templates (score < 0.3)"
            echo "  --condor     Submit nuisance fits to HTCondor"
            echo "  --parallel   Number of parallel local jobs [default: 4]"
            echo "  --skip-initial  Skip initial fit (use existing)"
            echo "  --skip-fits  Skip nuisance fits (use existing)"
            echo "  --plot-only  Only generate impact plot from existing json"
            echo "  --dry-run    Print commands without executing"
            echo "  --verbose    Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era, --channel, and --masspoint are required"
    exit 1
fi

# Get WORKDIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Template directory
BINNING_SUFFIX="${BINNING}"
if [[ "$PARTIAL_UNBLIND" == true ]]; then
    BINNING_SUFFIX="${BINNING}_partial_unblind"
fi
TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV1/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING_SUFFIX}"

# Check if template directory exists
if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "ERROR: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi

# Create output directory
OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/impacts"
mkdir -p "$OUTPUT_DIR"

# Log function
log() {
    if [[ "$VERBOSE" == true ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    fi
}

# Run function (respects dry-run)
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $1"
    else
        log "Running: $1"
        eval "$1"
    fi
}

# Change to template directory
cd "$TEMPLATE_DIR"
log "Working directory: $(pwd)"

# Clean up working directory if running step 1 or step 2
if [[ "$DO_INITIAL" == true || "$DO_FITS" == true ]]; then
    echo "Cleaning up working directory..."
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] rm -f higgsCombine*.root"
        echo "[DRY-RUN] rm -rf ${OUTPUT_DIR}"
    else
        rm -f higgsCombine*.root
        rm -rf "${OUTPUT_DIR}"
        mkdir -p "$OUTPUT_DIR"
    fi
fi

# Set parameter range - wider for partial-unblind due to weaker constraints
if [[ "$PARTIAL_UNBLIND" == true ]]; then
    R_RANGE="r=-5,5"
else
    R_RANGE="r=-1,1"
fi

echo "Running Impacts for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})..."
echo "  Condor: ${CONDOR}"
echo "  Parallel jobs: ${PARALLEL}"
echo "  Parameter range: ${R_RANGE}"
echo ""

# Create workspace if needed
if [[ ! -f "workspace.root" ]] || [[ "datacard.txt" -nt "workspace.root" ]]; then
    echo "Creating workspace..."
    run_cmd "text2workspace.py datacard.txt -o workspace.root"
fi

# Step 1: Initial fit
if [[ "$DO_INITIAL" == true ]]; then
    echo ""
    echo "Step 1: Running initial fit..."
    run_cmd "combineTool.py -M Impacts \
        -d workspace.root \
        -m 120 \
        --doInitialFit \
        --robustFit 1 \
        -t -1 \
        --expectSignal 0 \
        --setParameterRanges ${R_RANGE} \
        -n .${MASSPOINT}.${METHOD}.${BINNING_SUFFIX} \
        2>&1 | tee ${OUTPUT_DIR}/initial_fit.out"
    # Note: Don't move files yet - Step 2 needs the initial fit file in the current directory
fi

# Step 2: Run fits for each nuisance parameter
if [[ "$DO_FITS" == true ]]; then
    echo ""
    echo "Step 2: Running nuisance parameter fits..."

    if [[ "$CONDOR" == true ]]; then
        run_cmd "combineTool.py -M Impacts \
            -d workspace.root \
            -m 120 \
            --doFits \
            --robustFit 1 \
            -t -1 \
            --setParameterRanges ${R_RANGE} \
            --expectSignal 0 \
            -n .${MASSPOINT}.${METHOD}.${BINNING_SUFFIX} \
            --job-mode condor \
            --task-name impacts_${MASSPOINT} \
            2>&1 | tee ${OUTPUT_DIR}/nuisance_fits.out"

        if [[ "$DRY_RUN" == false ]]; then
            echo ""
            echo "Jobs submitted to HTCondor."
            echo "Monitor with: condor_q"
            echo "After completion, run with --skip-initial --skip-fits to generate plots."
            exit 0
        fi
    else
        run_cmd "combineTool.py -M Impacts \
            -d workspace.root \
            -m 120 \
            --doFits \
            --robustFit 1 \
            -t -1 \
            --setParameterRanges ${R_RANGE} \
            --expectSignal 0 \
            -n .${MASSPOINT}.${METHOD}.${BINNING_SUFFIX} \
            --parallel ${PARALLEL} \
            2>&1 | tee ${OUTPUT_DIR}/nuisance_fits.out"
        # Note: Don't move files yet - Step 3 (collect) needs them in the current directory
    fi
fi

# Step 3: Collect impacts and generate plot
if [[ "$DO_PLOT" == true ]]; then
    echo ""
    echo "Step 3: Collecting impacts..."

    run_cmd "combineTool.py -M Impacts \
        -d workspace.root \
        -m 120 \
        -n .${MASSPOINT}.${METHOD}.${BINNING_SUFFIX} \
        -o ${OUTPUT_DIR}/impacts.json \
        2>&1 | tee ${OUTPUT_DIR}/collect.out"

    # Move all higgsCombine output files to OUTPUT_DIR after collect step
    if [[ "$DRY_RUN" == false ]]; then
        mv -f higgsCombine*.root "${OUTPUT_DIR}/" 2>/dev/null || true
    fi

    echo ""
    echo "Step 4: Generating impact plot..."
    run_cmd "plotImpacts.py \
        -i ${OUTPUT_DIR}/impacts.json \
        -o ${OUTPUT_DIR}/impacts \
        2>&1 | tee ${OUTPUT_DIR}/plot.out"

    if [[ "$DRY_RUN" == false ]]; then
        if [[ -f "${OUTPUT_DIR}/impacts.pdf" ]]; then
            echo ""
            echo "SUCCESS: Impacts saved to ${OUTPUT_DIR}/"
            echo ""
            echo "Output files:"
            ls -la "${OUTPUT_DIR}/impacts."*
        else
            echo "WARNING: Impact plot not generated"
        fi
    fi
fi

echo ""
echo "Done."
