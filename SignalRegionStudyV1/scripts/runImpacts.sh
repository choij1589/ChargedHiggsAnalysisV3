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
    R_RANGE="r=-50,50"
else
    R_RANGE="r=-1,1"
fi

# Build fit options: use Asimov (expected) for blinded, real data for unblinded
if [[ "$PARTIAL_UNBLIND" == true ]]; then
    # Unblinded: use real data_obs from shapes.root
    ASIMOV_OPTIONS=""
else
    # Blinded: use Asimov dataset with background-only hypothesis
    ASIMOV_OPTIONS="-t -1 --expectSignal 0"
fi

echo "Running Impacts for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})..."
echo "  Condor: ${CONDOR}"
echo "  Parallel jobs: ${PARALLEL}"
echo "  Parameter range: ${R_RANGE}"
echo "  Data mode: $(if [[ -n "$ASIMOV_OPTIONS" ]]; then echo 'Asimov (expected)'; else echo 'Observed (real data)'; fi)"
echo ""

# ============================================================
# Condor DAG workflow: Run entire pipeline in HTCondor
# ============================================================
if [[ "$CONDOR" == true ]]; then
    echo "Preparing DAG workflow for HTCondor..."

    # Create condor directory
    CONDOR_DIR="${OUTPUT_DIR}/condor"
    mkdir -p "${CONDOR_DIR}/logs"

    # CMSSW path for Combine
    CMSSW_BASE="${WORKDIR}/SignalRegionStudyV1/CMSSW_14_1_0_pre4/src"

    # Suffix for output file naming
    SUFFIX="${MASSPOINT}.${METHOD}.${BINNING_SUFFIX}"

    # Extract nuisance parameters from datacard
    # Datacard structure: comments, shapes section, bin/obs section, process/rate section, then systematics
    # Separators: 1st after shapes, 2nd after bin/obs, 3rd after shapes header, 4th after rate
    # Systematics start after the 4th separator line
    echo "Extracting nuisance parameters from datacard..."
    NUISANCES=()
    separator_count=0
    while IFS= read -r line; do
        # Count separator lines (lines starting with dashes)
        if [[ "$line" =~ ^-+ ]]; then
            separator_count=$((separator_count + 1))
            continue
        fi
        # After the 4th separator (after rate line), we're in systematics
        if [[ $separator_count -ge 4 && -n "$line" && ! "$line" =~ ^# && ! "$line" =~ autoMCStats ]]; then
            nuisance=$(echo "$line" | awk '{print $1}')
            if [[ -n "$nuisance" ]]; then
                NUISANCES+=("$nuisance")
            fi
        fi
    done < datacard.txt
    echo "Found ${#NUISANCES[@]} nuisance parameters"

    # ========== workspace.sh ==========
    cat > "${CONDOR_DIR}/workspace.sh" << EOFWORKSPACE
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
text2workspace.py datacard.txt -o workspace.root
EOFWORKSPACE
    chmod +x "${CONDOR_DIR}/workspace.sh"

    # ========== workspace.sub ==========
    WORKSPACE_INPUT_FILES="${TEMPLATE_DIR}/datacard.txt"
    for shapefile in "${TEMPLATE_DIR}"/shapes*.root; do
        if [[ -f "$shapefile" ]]; then
            WORKSPACE_INPUT_FILES="${WORKSPACE_INPUT_FILES},${shapefile}"
        fi
    done

    cat > "${CONDOR_DIR}/workspace.sub" << EOF
universe = vanilla
executable = workspace.sh
output = logs/workspace.out
error = logs/workspace.err
log = impacts.log

request_cpus = 1
request_memory = 2GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = ${WORKSPACE_INPUT_FILES}
transfer_output_files = workspace.root
when_to_transfer_output = ON_EXIT

#+SingularityImage = "/data9/Users/choij/Singularity/images/private-el8.sif"
#+SingularityBind = "/cvmfs,/cms,/share"

queue
EOF

    # ========== initial_fit.sh ==========
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

    # ========== initial_fit.sub ==========
    cat > "${CONDOR_DIR}/initial_fit.sub" << EOF
universe = vanilla
executable = initial_fit.sh
output = logs/initial_fit.out
error = logs/initial_fit.err
log = impacts.log

request_cpus = 1
request_memory = 4GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = workspace.root
transfer_output_files = higgsCombine_initialFit_.${SUFFIX}.MultiDimFit.mH120.root
when_to_transfer_output = ON_EXIT

#+SingularityImage = "/data9/Users/choij/Singularity/images/private-el8.sif"
#+SingularityBind = "/cvmfs,/cms,/share"

queue
EOF

    # ========== fit_nuisance.sh ==========
    cat > "${CONDOR_DIR}/fit_nuisance.sh" << EOFFIT
#!/bin/bash
NUISANCE=\$1
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
combineTool.py -M Impacts \\
    -d workspace.root \\
    -m 120 \\
    --doFits \\
    --robustFit 1 \\
    ${ASIMOV_OPTIONS} \\
    --setParameterRanges ${R_RANGE} \\
    -n .${SUFFIX} \\
    --named \${NUISANCE}
EOFFIT
    chmod +x "${CONDOR_DIR}/fit_nuisance.sh"

    # ========== fit_nuisance.sub ==========
    cat > "${CONDOR_DIR}/fit_nuisance.sub" << EOF
universe = vanilla
executable = fit_nuisance.sh
arguments = \$(nuisance)
output = logs/fit_\$(nuisance).out
error = logs/fit_\$(nuisance).err
log = impacts.log

request_cpus = 1
request_memory = 4GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = workspace.root,higgsCombine_initialFit_.${SUFFIX}.MultiDimFit.mH120.root
transfer_output_files = higgsCombine_paramFit_\$(nuisance).${SUFFIX}.MultiDimFit.mH120.root
when_to_transfer_output = ON_EXIT

#+SingularityImage = "/data9/Users/choij/Singularity/images/private-el8.sif"
#+SingularityBind = "/cvmfs,/cms,/share"

queue
EOF

    # ========== collect.sh ==========
    cat > "${CONDOR_DIR}/collect.sh" << EOFCOLLECT
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
combineTool.py -M Impacts \\
    -d workspace.root \\
    -m 120 \\
    -n .${SUFFIX} \\
    -o impacts.json
EOFCOLLECT
    chmod +x "${CONDOR_DIR}/collect.sh"

    # ========== collect.sub ==========
    # Build the list of all fit output files for transfer
    COLLECT_INPUT_FILES="workspace.root,higgsCombine_initialFit_.${SUFFIX}.MultiDimFit.mH120.root"
    for nuisance in "${NUISANCES[@]}"; do
        COLLECT_INPUT_FILES="${COLLECT_INPUT_FILES},higgsCombine_paramFit_${nuisance}.${SUFFIX}.MultiDimFit.mH120.root"
    done

    cat > "${CONDOR_DIR}/collect.sub" << EOF
universe = vanilla
executable = collect.sh
output = logs/collect.out
error = logs/collect.err
log = impacts.log

request_cpus = 1
request_memory = 2GB
request_disk = 1GB

should_transfer_files = YES
transfer_input_files = ${COLLECT_INPUT_FILES}
transfer_output_files = impacts.json
when_to_transfer_output = ON_EXIT

#+SingularityImage = "/data9/Users/choij/Singularity/images/private-el8.sif"
#+SingularityBind = "/cvmfs,/cms,/share"

queue
EOF

    # ========== plot.sh ==========
    cat > "${CONDOR_DIR}/plot.sh" << EOFPLOT
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
plotImpacts.py -i impacts.json -o impacts
EOFPLOT
    chmod +x "${CONDOR_DIR}/plot.sh"

    # ========== plot.sub ==========
    cat > "${CONDOR_DIR}/plot.sub" << EOF
universe = vanilla
executable = plot.sh
output = logs/plot.out
error = logs/plot.err
log = impacts.log

request_cpus = 1
request_memory = 2GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = impacts.json
transfer_output_files = impacts.pdf,impacts.png
when_to_transfer_output = ON_EXIT

#+SingularityImage = "/data9/Users/choij/Singularity/images/private-el8.sif"
#+SingularityBind = "/cvmfs,/cms,/share"

queue
EOF

    # ========== impacts.dag ==========
    DAG_FILE="${CONDOR_DIR}/impacts.dag"
    {
        echo "# Impacts DAG workflow"
        echo "# Generated for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})"
        echo ""

        # Step 1: Create workspace
        echo "# Step 1: Create workspace"
        echo "JOB workspace workspace.sub"
        echo ""

        # Step 2: Initial fit
        echo "# Step 2: Initial fit"
        echo "JOB initial_fit initial_fit.sub"
        echo ""

        # Step 3: Nuisance parameter fits
        echo "# Step 3: Nuisance parameter fits (${#NUISANCES[@]} jobs)"
        FIT_JOBS=""
        for nuisance in "${NUISANCES[@]}"; do
            job_name="fit_${nuisance}"
            echo "JOB ${job_name} fit_nuisance.sub"
            echo "VARS ${job_name} nuisance=\"${nuisance}\""
            FIT_JOBS="${FIT_JOBS} ${job_name}"
        done
        echo ""

        # Step 4: Collect results
        echo "# Step 4: Collect impacts"
        echo "JOB collect collect.sub"
        echo ""

        # Step 5: Generate plot
        echo "# Step 5: Plot impacts"
        echo "JOB plot plot.sub"
        echo ""

        # Dependencies
        echo "# Dependencies"
        echo "PARENT workspace CHILD initial_fit"
        echo "PARENT initial_fit CHILD${FIT_JOBS}"
        echo "PARENT${FIT_JOBS} CHILD collect"
        echo "PARENT collect CHILD plot"

    } > "$DAG_FILE"

    echo ""
    echo "=========================================="
    echo "DAG workflow prepared in:"
    echo "  ${CONDOR_DIR}"
    echo ""
    echo "Files created:"
    echo "  impacts.dag       - DAG workflow definition"
    echo "  workspace.sub     - Workspace creation job"
    echo "  initial_fit.sub   - Initial fit job"
    echo "  fit_nuisance.sub  - Nuisance fit job template (${#NUISANCES[@]} jobs)"
    echo "  collect.sub       - Collect results job"
    echo "  plot.sub          - Plot generation job"
    echo "  *.sh              - Wrapper scripts"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] To submit:"
        echo "  cd ${CONDOR_DIR}"
        echo "  condor_submit_dag impacts.dag"
    else
        echo "Submitting DAG workflow..."
        cd "${CONDOR_DIR}"
        condor_submit_dag impacts.dag
        echo ""
        echo "Monitor with: condor_q -dag"
        echo "DAG log: ${CONDOR_DIR}/impacts.dag.dagman.out"
    fi
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
