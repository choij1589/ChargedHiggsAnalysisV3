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
CONDOR=false
PARALLEL=4
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
            echo "  --binning    Binning scheme (uniform, sigma) [default: uniform]"
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
TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV1/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING}"

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

echo "Running Impacts for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING})..."
echo "  Condor: ${CONDOR}"
echo "  Parallel jobs: ${PARALLEL}"
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
        --setParameterRanges r=-1,1 \
        -n .${MASSPOINT}.${METHOD}.${BINNING} \
        2>&1 | tee ${OUTPUT_DIR}/initial_fit.out"

    if [[ "$DRY_RUN" == false ]]; then
        mv -f higgsCombine*.root "${OUTPUT_DIR}/" 2>/dev/null || true
    fi
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
            --expectSignal 0 \
            --setParameterRanges r=-1,1 \
            -n .${MASSPOINT}.${METHOD}.${BINNING} \
            --job-mode condor \
            --sub-opts '+JobFlavour = \"longlunch\"' \
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
            --expectSignal 0 \
            --setParameterRanges r=-1,1 \
            -n .${MASSPOINT}.${METHOD}.${BINNING} \
            --parallel ${PARALLEL} \
            2>&1 | tee ${OUTPUT_DIR}/nuisance_fits.out"

        if [[ "$DRY_RUN" == false ]]; then
            mv -f higgsCombine*.root "${OUTPUT_DIR}/" 2>/dev/null || true
        fi
    fi
fi

# Step 3: Collect impacts and generate plot
if [[ "$DO_PLOT" == true ]]; then
    echo ""
    echo "Step 3: Collecting impacts..."

    # Move any remaining output files
    if [[ "$DRY_RUN" == false ]]; then
        mv -f higgsCombine*.root "${OUTPUT_DIR}/" 2>/dev/null || true
    fi

    run_cmd "combineTool.py -M Impacts \
        -d workspace.root \
        -m 120 \
        -n .${MASSPOINT}.${METHOD}.${BINNING} \
        -o ${OUTPUT_DIR}/impacts.json \
        2>&1 | tee ${OUTPUT_DIR}/collect.out"

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
