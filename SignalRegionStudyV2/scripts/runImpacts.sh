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
EXPECT_SIGNAL=1  # Default: inject signal (use 0 for background-only)
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
        --expect-signal)
            EXPECT_SIGNAL="$2"
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
            echo "  --binning    Binning scheme (uniform, extended) [default: uniform]"
            echo "  --partial-unblind  Use partial-unblind templates (score < 0.3)"
            echo "  --expect-signal N  Expected signal strength for Asimov (0 or 1) [default: 1]"
            echo "  --condor     Run full workflow in HTCondor via DAG (all steps)"
            echo "  --parallel   Number of parallel local jobs [default: 16]"
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
TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV2/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING_SUFFIX}"

# Check if template directory exists
if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "ERROR: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi

# Create output directory (include expectSignal in name to avoid overwriting)
if [[ "$PARTIAL_UNBLIND" == true ]]; then
    OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/impacts_obs"
else
    OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/impacts_r${EXPECT_SIGNAL}"
fi
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
    R_RANGE="r=-50,50"
else
    R_RANGE="r=-5,5"
fi

# Build fit options: use Asimov (expected) for blinded, real data for unblinded
if [[ "$PARTIAL_UNBLIND" == true ]]; then
    # Unblinded: use real data_obs from shapes.root
    ASIMOV_OPTIONS=""
else
    # Blinded: use Asimov dataset
    ASIMOV_OPTIONS="-t -1 --expectSignal ${EXPECT_SIGNAL}"
fi

echo "Running Impacts for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})..."
echo "  Condor: ${CONDOR}"
echo "  Parallel jobs: ${PARALLEL}"
echo "  Parameter range: ${R_RANGE}"
echo "  Data mode: $(if [[ -n "$ASIMOV_OPTIONS" ]]; then echo "Asimov (expectSignal=${EXPECT_SIGNAL})"; else echo 'Observed (real data)'; fi)"
echo ""

# ============================================================
# Condor DAG workflow: Run entire pipeline in HTCondor
# Each nuisance parameter gets its own condor job
# ============================================================
if [[ "$CONDOR" == true ]]; then
    echo "Preparing per-nuisance DAG workflow for HTCondor..."

    # Create condor directory
    CONDOR_DIR="${OUTPUT_DIR}/condor"
    mkdir -p "${CONDOR_DIR}/logs"

    # CMSSW path for Combine
    CMSSW_BASE="${WORKDIR}/Common/CMSSW_14_1_0_pre4/src"

    # Suffix for output file naming
    SUFFIX="${MASSPOINT}.${METHOD}.${BINNING_SUFFIX}"

    # ========== Step 1: Create workspace locally ==========
    echo "Creating workspace locally..."
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] text2workspace.py datacard.txt -o workspace.root"
        echo "[DRY-RUN] Skipping workspace creation - cannot extract nuisance parameters"
        echo "[DRY-RUN] Would create individual fit jobs for each nuisance parameter"
        exit 0
    else
        text2workspace.py datacard.txt -o workspace.root
    fi

    # ========== Step 2: Extract nuisance parameters from workspace ==========
    echo "Extracting nuisance parameters from workspace..."
    NUISANCES=$(python3 << 'PYEOF'
import ROOT
ROOT.gROOT.SetBatch(True)
f = ROOT.TFile("workspace.root")
w = f.Get("w")
mc = w.genobj("ModelConfig")
nuisances = mc.GetNuisanceParameters()
for p in nuisances:
    print(p.GetName())
f.Close()
PYEOF
)

    # Convert to array
    readarray -t NUIS_ARRAY <<< "$NUISANCES"
    NUM_NUISANCES=${#NUIS_ARRAY[@]}
    echo "Found ${NUM_NUISANCES} nuisance parameters"

    if [[ $NUM_NUISANCES -eq 0 ]]; then
        echo "ERROR: No nuisance parameters found in workspace"
        exit 1
    fi

    # ========== Step 3: Generate initial_fit job ==========
    echo "Generating initial_fit job..."
    cat > "${CONDOR_DIR}/initial_fit.sh" << EOFINITIAL
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
combineTool.py -M Impacts \\
    -d workspace.root \\
    -m 120 \\
    --doInitialFit \\
    --robustFit 1 \\
    ${ASIMOV_OPTIONS} \\
    --setParameterRanges ${R_RANGE} \\
    -n .${SUFFIX}
EOFINITIAL
    chmod +x "${CONDOR_DIR}/initial_fit.sh"

    cat > "${CONDOR_DIR}/initial_fit.sub" << EOF
universe = vanilla
executable = ${CONDOR_DIR}/initial_fit.sh
output = ${CONDOR_DIR}/logs/initial_fit.out
error = ${CONDOR_DIR}/logs/initial_fit.err
log = ${CONDOR_DIR}/impacts.log

request_cpus = 1
request_memory = 2GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = ${CONDOR_DIR}/workspace.root
transfer_output_files = higgsCombine_initialFit_.${SUFFIX}.MultiDimFit.mH120.root
when_to_transfer_output = ON_EXIT

queue
EOF

    # ========== Step 4: Generate individual fit jobs ==========
    echo "Generating ${NUM_NUISANCES} individual fit jobs..."
    FIT_JOB_NAMES=()

    for NUIS in "${NUIS_ARRAY[@]}"; do
        # Sanitize nuisance name for filename (replace special chars)
        NUIS_SAFE=$(echo "$NUIS" | tr '/' '_' | tr ':' '_')
        FIT_JOB_NAMES+=("fit_${NUIS_SAFE}")

        cat > "${CONDOR_DIR}/logs/fit_${NUIS_SAFE}.sh" << EOFFIT
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
echo "Running combineTool.py for ${NUIS}..."
combineTool.py -M Impacts \\
    -d workspace.root \\
    -m 120 \\
    --doFits \\
    --robustFit 1 \\
    ${ASIMOV_OPTIONS} \\
    --setParameterRanges ${R_RANGE} \\
    -n .${SUFFIX} \\
    --named ${NUIS}
echo "Combine finished. Files created:"
ls -la higgsCombine*.root 2>/dev/null || echo "No higgsCombine*.root files found!"
EOFFIT
        chmod +x "${CONDOR_DIR}/logs/fit_${NUIS_SAFE}.sh"

        cat > "${CONDOR_DIR}/logs/fit_${NUIS_SAFE}.sub" << EOFSUB
universe = vanilla
executable = ${CONDOR_DIR}/logs/fit_${NUIS_SAFE}.sh
output = ${CONDOR_DIR}/logs/fit_${NUIS_SAFE}.out
error = ${CONDOR_DIR}/logs/fit_${NUIS_SAFE}.err
log = ${CONDOR_DIR}/impacts.log

request_cpus = 1
request_memory = 2GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = ${CONDOR_DIR}/workspace.root,${CONDOR_DIR}/higgsCombine_initialFit_.${SUFFIX}.MultiDimFit.mH120.root
transfer_output_files = higgsCombine_paramFit_.${SUFFIX}_${NUIS}.MultiDimFit.mH120.root
when_to_transfer_output = ON_EXIT

queue
EOFSUB
    done

    # ========== Step 5: Generate collect_plot job ==========
    echo "Generating collect_plot job..."

    # Build list of all fit output files for transfer (use absolute paths)
    # All fit outputs are in condor/ (DAG transfers outputs to DAG submission directory)
    FIT_OUTPUT_FILES="${CONDOR_DIR}/higgsCombine_initialFit_.${SUFFIX}.MultiDimFit.mH120.root"
    for NUIS in "${NUIS_ARRAY[@]}"; do
        FIT_OUTPUT_FILES="${FIT_OUTPUT_FILES},${CONDOR_DIR}/higgsCombine_paramFit_.${SUFFIX}_${NUIS}.MultiDimFit.mH120.root"
    done

    cat > "${CONDOR_DIR}/collect_plot.sh" << EOFCOLLECT
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

echo "Collecting impacts..."
combineTool.py -M Impacts \\
    -d workspace.root \\
    -m 120 \\
    -n .${SUFFIX} \\
    -o impacts.json

echo "Filtering stat bin nuisances..."
python3 filterImpacts.py -i impacts.json -o impacts_filtered.json

echo "Generating impact plots..."
plotImpacts.py -i impacts.json -o impacts
plotImpacts.py -i impacts_filtered.json -o impacts_filtered --summary

echo "Done!"
EOFCOLLECT
    chmod +x "${CONDOR_DIR}/collect_plot.sh"

    cat > "${CONDOR_DIR}/collect_plot.sub" << EOF
universe = vanilla
executable = ${CONDOR_DIR}/collect_plot.sh
output = ${CONDOR_DIR}/logs/collect_plot.out
error = ${CONDOR_DIR}/logs/collect_plot.err
log = ${CONDOR_DIR}/impacts.log

request_cpus = 1
request_memory = 2GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = ${CONDOR_DIR}/workspace.root,${CONDOR_DIR}/filterImpacts.py,${FIT_OUTPUT_FILES}
transfer_output_files = impacts.json,impacts.pdf,impacts_filtered.json,impacts_filtered.pdf,impacts_filtered_summary.pdf
when_to_transfer_output = ON_EXIT

queue
EOF

    # ========== Step 6: Generate cleanup script ==========
    echo "Generating cleanup script..."
    cat > "${CONDOR_DIR}/cleanup.sh" << EOFCLEANUP
#!/bin/bash
# Move higgsCombine*.root files to logs/ directory for organization
cd "${CONDOR_DIR}"
mv -f higgsCombine*.root logs/ 2>/dev/null || true
echo "Moved higgsCombine*.root files to logs/"
EOFCLEANUP
    chmod +x "${CONDOR_DIR}/cleanup.sh"

    # ========== Step 7: Generate DAG file ==========
    echo "Generating DAG file..."
    DAG_FILE="${CONDOR_DIR}/impacts.dag"
    {
        echo "# Impacts DAG workflow with per-nuisance jobs"
        echo "# Generated for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})"
        echo "# Total jobs: 1 initial_fit + ${NUM_NUISANCES} fit jobs + 1 collect_plot = $((NUM_NUISANCES + 2))"
        echo ""

        # Initial fit job
        echo "# Initial fit"
        echo "JOB initial_fit initial_fit.sub"
        echo ""

        # Individual fit jobs
        echo "# Individual nuisance parameter fits (${NUM_NUISANCES} jobs)"
        for JOB_NAME in "${FIT_JOB_NAMES[@]}"; do
            echo "JOB ${JOB_NAME} logs/${JOB_NAME}.sub"
        done
        echo ""

        # Collect and plot job
        echo "# Collect results and plot"
        echo "JOB collect_plot collect_plot.sub"
        echo ""

        # Dependencies: initial_fit -> all fit jobs -> collect_plot
        echo "# Dependencies"
        echo "PARENT initial_fit CHILD ${FIT_JOB_NAMES[*]}"
        echo "PARENT ${FIT_JOB_NAMES[*]} CHILD collect_plot"
        echo ""

        # Post-script: cleanup after collect_plot completes
        echo "# Cleanup: move .root files to logs/ directory"
        echo "SCRIPT POST collect_plot cleanup.sh"

    } > "$DAG_FILE"

    # Copy workspace and filterImpacts.py to condor directory for transfer
    cp workspace.root "${CONDOR_DIR}/"
    cp "${SCRIPT_DIR}/../python/filterImpacts.py" "${CONDOR_DIR}/"

    echo ""
    echo "=========================================="
    echo "DAG workflow prepared in:"
    echo "  ${CONDOR_DIR}"
    echo ""
    echo "Jobs created:"
    echo "  1 initial_fit job"
    echo "  ${NUM_NUISANCES} individual nuisance fit jobs"
    echo "  1 collect_plot job"
    echo "  Total: $((NUM_NUISANCES + 2)) jobs"
    echo ""

    echo "Submitting DAG workflow..."
    cd "${CONDOR_DIR}"
    condor_submit_dag impacts.dag
    echo ""
    echo "Monitor with: condor_q -dag"
    echo "DAG log: ${CONDOR_DIR}/impacts.dag.dagman.out"
    echo "=========================================="
    exit 0
fi

# ============================================================
# Local execution: Run steps sequentially
# ============================================================

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
        ${ASIMOV_OPTIONS} \
        --setParameterRanges ${R_RANGE} \
        -n .${MASSPOINT}.${METHOD}.${BINNING_SUFFIX} \
        2>&1 | tee ${OUTPUT_DIR}/initial_fit.out"
    # Note: Don't move files yet - Step 2 needs the initial fit file in the current directory
fi

# Step 2: Run fits for each nuisance parameter
if [[ "$DO_FITS" == true ]]; then
    echo ""
    echo "Step 2: Running nuisance parameter fits..."
    run_cmd "combineTool.py -M Impacts \
        -d workspace.root \
        -m 120 \
        --doFits \
        --robustFit 1 \
        ${ASIMOV_OPTIONS} \
        --setParameterRanges ${R_RANGE} \
        -n .${MASSPOINT}.${METHOD}.${BINNING_SUFFIX} \
        --parallel ${PARALLEL} \
        2>&1 | tee ${OUTPUT_DIR}/nuisance_fits.out"
    # Note: Don't move files yet - Step 3 (collect) needs them in the current directory
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
    echo "Step 4: Filtering stat bin nuisances..."
    FILTER_SCRIPT="${SCRIPT_DIR}/../python/filterImpacts.py"
    run_cmd "python3 ${FILTER_SCRIPT} \
        -i ${OUTPUT_DIR}/impacts.json \
        -o ${OUTPUT_DIR}/impacts_filtered.json \
        2>&1 | tee ${OUTPUT_DIR}/filter.out"

    echo ""
    echo "Step 5: Generating impact plots..."
    # Full impacts plot
    run_cmd "plotImpacts.py \
        -i ${OUTPUT_DIR}/impacts.json \
        -o ${OUTPUT_DIR}/impacts \
        2>&1 | tee ${OUTPUT_DIR}/plot.out"

    # Filtered impacts plot with summary
    run_cmd "plotImpacts.py \
        -i ${OUTPUT_DIR}/impacts_filtered.json \
        -o ${OUTPUT_DIR}/impacts_filtered \
        --summary \
        2>&1 | tee -a ${OUTPUT_DIR}/plot.out"

    if [[ "$DRY_RUN" == false ]]; then
        if [[ -f "${OUTPUT_DIR}/impacts.pdf" ]]; then
            echo ""
            echo "SUCCESS: Impacts saved to ${OUTPUT_DIR}/"
            echo ""
            echo "Output files:"
            ls -la "${OUTPUT_DIR}/impacts."* "${OUTPUT_DIR}/impacts_filtered."* 2>/dev/null
        else
            echo "WARNING: Impact plot not generated"
        fi
    fi
fi

echo ""
echo "Done."
