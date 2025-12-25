#!/bin/bash
#
# runHybridNew.sh - Run HybridNew limit calculation with toys
#
# Supports both local execution and HTCondor batch submission.
#
# Usage:
#   ./runHybridNew.sh --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
#   ./runHybridNew.sh --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform --condor
#

set -e

# Default values
ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
BINNING="uniform"
CONDOR=false
NTOYS=500
NJOBS=10
RMIN=0.0
RMAX=2.0
RSTEP=0.2
DRY_RUN=false
VERBOSE=false
MERGE_ONLY=false
EXTRACT_ONLY=false
AUTO_GRID=false

# Quantiles for expected limits
QUANTILES=(0.025 0.160 0.500 0.840 0.975)

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
        --ntoys)
            NTOYS="$2"
            shift 2
            ;;
        --njobs)
            NJOBS="$2"
            shift 2
            ;;
        --rmin)
            RMIN="$2"
            shift 2
            ;;
        --rmax)
            RMAX="$2"
            shift 2
            ;;
        --rstep)
            RSTEP="$2"
            shift 2
            ;;
        --merge-only)
            MERGE_ONLY=true
            shift
            ;;
        --extract-only)
            EXTRACT_ONLY=true
            shift
            ;;
        --auto-grid)
            AUTO_GRID=true
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
            echo "  --condor     Submit jobs to HTCondor"
            echo "  --ntoys      Number of toys per job [default: 500]"
            echo "  --njobs      Number of parallel jobs [default: 10]"
            echo "  --rmin       Minimum r value [default: 0.01]"
            echo "  --rmax       Maximum r value [default: 5.0]"
            echo "  --rstep      r value step size [default: 0.5]"
            echo "  --merge-only Only merge existing toy results"
            echo "  --extract-only Only extract limits from merged grid"
            echo "  --auto-grid  Auto-tune grid from Asymptotic results (overrides rmin/rmax/rstep)"
            echo "  --dry-run    Prepare DAG files without submitting (with --condor)"
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

# Create output directories
OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/hybridnew"
CONDOR_DIR="${OUTPUT_DIR}/condor"
TOYS_DIR="${OUTPUT_DIR}/toys"

if [[ "$CONDOR" == true ]]; then
    # Clean up existing output directory for fresh condor submission
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "Cleaning up existing output directory: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    fi
    mkdir -p "$OUTPUT_DIR" "$CONDOR_DIR"
else
    mkdir -p "$OUTPUT_DIR" "$TOYS_DIR"
fi

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

# Auto-tune grid from Asymptotic results
if [[ "$AUTO_GRID" == true ]]; then
    echo "Auto-tuning grid from Asymptotic results..."

    ASYMPTOTIC_DIR="${TEMPLATE_DIR}/combine_output/asymptotic"
    ASYMPTOTIC_FILE=""

    # Find asymptotic result file
    for pattern in "higgsCombine.*.AsymptoticLimits.mH120.root" "higgsCombineTest.AsymptoticLimits.mH120.root"; do
        found=$(ls "${ASYMPTOTIC_DIR}"/${pattern} 2>/dev/null | head -1)
        if [[ -n "$found" ]]; then
            ASYMPTOTIC_FILE="$found"
            break
        fi
    done

    if [[ -z "$ASYMPTOTIC_FILE" ]]; then
        echo "ERROR: No Asymptotic results found in ${ASYMPTOTIC_DIR}"
        echo "Run AsymptoticLimits first, then use --auto-grid"
        exit 1
    fi

    echo "  Reading from: ${ASYMPTOTIC_FILE}"

    # Parse asymptotic limits and calculate optimal grid
    read RMIN RMAX RSTEP < <(python3 << EOFPYTHON
import ROOT
ROOT.gROOT.SetBatch(True)

f = ROOT.TFile.Open("${ASYMPTOTIC_FILE}")
tree = f.Get("limit")

limits = []
for i in range(tree.GetEntries()):
    tree.GetEntry(i)
    limits.append(tree.limit)

f.Close()

# limits order: exp-2, exp-1, exp0, exp+1, exp+2, obs
if len(limits) >= 5:
    exp_minus2 = limits[0]  # 2.5% quantile
    exp_plus2 = limits[4]   # 97.5% quantile
    exp_median = limits[2]  # 50% quantile

    # Grid design:
    # - rmin: 0.5x of exp-2, minimum 0.01
    # - rmax: 1.5x of exp+2
    # - rstep: ~5% of median, rounded to nice value

    rmin = max(0.01, exp_minus2 * 0.5)
    rmax = exp_plus2 * 1.5

    # Round rstep to nice values
    raw_step = exp_median * 0.1
    if raw_step < 0.02:
        rstep = 0.01
    elif raw_step < 0.05:
        rstep = 0.02
    elif raw_step < 0.1:
        rstep = 0.05
    elif raw_step < 0.2:
        rstep = 0.1
    else:
        rstep = 0.2

    # Round rmin down and rmax up to step size
    rmin = max(0.01, int(rmin / rstep) * rstep)
    rmax = (int(rmax / rstep) + 1) * rstep

    print(f"{rmin:.4f} {rmax:.4f} {rstep:.4f}")
else:
    print("0.01 2.0 0.1")  # fallback
EOFPYTHON
)

    echo "  Auto-tuned: rmin=${RMIN}, rmax=${RMAX}, rstep=${RSTEP}"
fi

echo "HybridNew limit calculation for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING})"
echo "  Toys per job: ${NTOYS}"
echo "  Number of jobs: ${NJOBS}"
echo "  r range: [${RMIN}, ${RMAX}] step ${RSTEP}"
echo "  HTCondor: ${CONDOR}"
echo ""

# Step 1: Create workspace (skip if --condor, will be done by DAG)
if [[ "$CONDOR" != true ]]; then
    if [[ ! -f "workspace.root" ]] || [[ "datacard.txt" -nt "workspace.root" ]]; then
        echo "Creating workspace..."
        run_cmd "text2workspace.py datacard.txt -o workspace.root"
    fi
fi

# Generate r values using Python (bc may not be available)
R_VALUES=($(python3 -c "
rmin, rmax, rstep = $RMIN, $RMAX, $RSTEP
r = rmin
values = []
while r <= rmax + 1e-9:  # small epsilon for floating point
    values.append(f'{r:.4f}')
    r += rstep
print(' '.join(values))
"))
echo "r values to scan: ${R_VALUES[*]}"

# If --condor, generate DAG files and submit (unless --dry-run)
if [[ "$CONDOR" == true ]]; then
    echo ""
    echo "Preparing DAG workflow..."

    # Create logs directory
    mkdir -p "${CONDOR_DIR}/logs"

    # Singularity configuration
    SINGULARITY_IMAGE="/data9/Users/choij/Singularity/images/private-el8.sif"

    # CMSSW path for Combine
    CMSSW_BASE="${WORKDIR}/SignalRegionStudyV1/CMSSW_14_1_0_pre4/src"

    # ========== workspace.sh ==========
    cat > "${CONDOR_DIR}/workspace.sh" << EOFWORKSPACE
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
text2workspace.py datacard.txt -o workspace.root
EOFWORKSPACE
    chmod +x "${CONDOR_DIR}/workspace.sh"

    # ========== workspace.sub ==========
    # Build list of input files (datacard + all shape files)
    WORKSPACE_INPUT_FILES="${TEMPLATE_DIR}/datacard.txt"
    for shapefile in "${TEMPLATE_DIR}"/shapes_*.root; do
        if [[ -f "$shapefile" ]]; then
            WORKSPACE_INPUT_FILES="${WORKSPACE_INPUT_FILES},${shapefile}"
        fi
    done

    cat > "${CONDOR_DIR}/workspace.sub" << EOF
universe = vanilla
executable = workspace.sh
output = logs/workspace.out
error = logs/workspace.err
log = hybridnew.log

request_cpus = 1
request_memory = 2GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = ${WORKSPACE_INPUT_FILES}
transfer_output_files = workspace.root
when_to_transfer_output = ON_EXIT

+SingularityImage = "${SINGULARITY_IMAGE}"
+SingularityBind = "/cvmfs,/cms,/share"

queue
EOF

    # ========== toy.sh ==========
    cat > "${CONDOR_DIR}/toy.sh" << EOFTOY
#!/bin/bash
R_VALUE=\$1
SEED=\$2

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

combine -M HybridNew workspace.root \\
    --LHCmode LHC-limits \\
    --singlePoint \${R_VALUE} \\
    --saveToys \\
    --saveHybridResult \\
    -T ${NTOYS} \\
    -s \${SEED} \\
    -n .r\${R_VALUE}.seed\${SEED}
EOFTOY
    chmod +x "${CONDOR_DIR}/toy.sh"

    # ========== toy.sub ==========
    cat > "${CONDOR_DIR}/toy.sub" << EOF
universe = vanilla
executable = toy.sh
arguments = \$(r_value) \$(seed)
output = logs/toy_\$(r_value)_\$(seed).out
error = logs/toy_\$(r_value)_\$(seed).err
log = hybridnew.log

request_cpus = 1
request_memory = 4GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = workspace.root
transfer_output_files = higgsCombine.r\$(r_value).seed\$(seed).HybridNew.mH120.\$(seed).root
when_to_transfer_output = ON_EXIT

+SingularityImage = "${SINGULARITY_IMAGE}"
+SingularityBind = "/cvmfs,/cms,/share"

queue
EOF

    # ========== merge.sh ==========
    cat > "${CONDOR_DIR}/merge.sh" << EOFMERGE
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

hadd -f hybridnew_grid.root higgsCombine.*.root
EOFMERGE
    chmod +x "${CONDOR_DIR}/merge.sh"

    # ========== merge.sub ==========
    # Build the list of toy output files for transfer
    TOY_FILES_LIST=""
    for r in "${R_VALUES[@]}"; do
        for ((j=0; j<NJOBS; j++)); do
            seed=$((1000 + j))
            TOY_FILES_LIST="${TOY_FILES_LIST}higgsCombine.r${r}.seed${seed}.HybridNew.mH120.${seed}.root,"
        done
    done
    TOY_FILES_LIST="${TOY_FILES_LIST%,}"  # Remove trailing comma

    cat > "${CONDOR_DIR}/merge.sub" << EOF
universe = vanilla
executable = merge.sh
output = logs/merge.out
error = logs/merge.err
log = hybridnew.log

request_cpus = 1
request_memory = 4GB
request_disk = 2GB

should_transfer_files = YES
transfer_input_files = ${TOY_FILES_LIST}
transfer_output_files = hybridnew_grid.root
when_to_transfer_output = ON_EXIT

+SingularityImage = "${SINGULARITY_IMAGE}"
+SingularityBind = "/cvmfs,/cms,/share"

queue
EOF

    # ========== extract.sh ==========
    cat > "${CONDOR_DIR}/extract.sh" << EOFEXTRACT
#!/bin/bash
QUANTILE=\$1
NAME=\$2

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

if [[ "\${QUANTILE}" != "observed" ]]; then
    # Expected limit
    combine -M HybridNew workspace.root \\
        --LHCmode LHC-limits \\
        --readHybridResults \\
        --grid=hybridnew_grid.root \\
        --expectedFromGrid \${QUANTILE} \\
        -n .\${NAME} \\
        -m 120 \\
        2>&1 | tee combine_\${NAME}.out
else
    # Observed limit
    combine -M HybridNew workspace.root \\
        --LHCmode LHC-limits \\
        --readHybridResults \\
        --grid=hybridnew_grid.root \\
        -n .\${NAME} \\
        -m 120 \\
        2>&1 | tee combine_\${NAME}.out
fi
EOFEXTRACT
    chmod +x "${CONDOR_DIR}/extract.sh"

    # ========== extract.sub ==========
    cat > "${CONDOR_DIR}/extract.sub" << EOF
universe = vanilla
executable = extract.sh
arguments = \$(quantile) \$(name)
output = logs/extract_\$(name).out
error = logs/extract_\$(name).err
log = hybridnew.log

request_cpus = 1
request_memory = 2GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = workspace.root,hybridnew_grid.root
transfer_output_files = combine_\$(name).out
when_to_transfer_output = ON_EXIT

+SingularityImage = "${SINGULARITY_IMAGE}"
+SingularityBind = "/cvmfs,/cms,/share"

queue
EOF

    # ========== hybridnew.dag ==========
    DAG_FILE="${CONDOR_DIR}/hybridnew.dag"

    {
        echo "# HybridNew DAG workflow"
        echo "# Generated for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING})"
        echo ""

        # Workspace job
        echo "# Step 1: Create workspace"
        echo "JOB workspace workspace.sub"
        echo ""

        # Toy generation jobs
        echo "# Step 2: Generate toys"
        TOY_JOBS=""
        for r in "${R_VALUES[@]}"; do
            for ((j=0; j<NJOBS; j++)); do
                seed=$((1000 + j))
                job_name="toy_r${r}_s${seed}"
                echo "JOB ${job_name} toy.sub"
                echo "VARS ${job_name} r_value=\"${r}\" seed=\"${seed}\""
                TOY_JOBS="${TOY_JOBS} ${job_name}"
            done
        done
        echo ""

        # Merge job
        echo "# Step 3: Merge toy results"
        echo "JOB merge merge.sub"
        echo ""

        # Extract jobs
        echo "# Step 4: Extract limits"
        EXTRACT_JOBS=""
        for q in "${QUANTILES[@]}"; do
            name="exp${q}"
            echo "JOB ${name} extract.sub"
            echo "VARS ${name} quantile=\"${q}\" name=\"${name}\""
            EXTRACT_JOBS="${EXTRACT_JOBS} ${name}"
        done
        echo "JOB observed extract.sub"
        echo "VARS observed quantile=\"observed\" name=\"obs\""
        EXTRACT_JOBS="${EXTRACT_JOBS} observed"
        echo ""

        # Dependencies
        echo "# Dependencies"
        echo "PARENT workspace CHILD${TOY_JOBS}"
        echo "PARENT${TOY_JOBS} CHILD merge"
        echo "PARENT merge CHILD${EXTRACT_JOBS}"

    } > "$DAG_FILE"

    echo ""
    echo "=========================================="
    echo "DAG workflow prepared in:"
    echo "  ${CONDOR_DIR}"
    echo ""
    echo "Files created:"
    echo "  hybridnew.dag  - DAG workflow definition"
    echo "  workspace.sub  - Workspace creation job"
    echo "  toy.sub        - Toy generation job template"
    echo "  merge.sub      - Merge job"
    echo "  extract.sub    - Limit extraction job template"
    echo "  *.sh           - Wrapper scripts"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] To submit:"
        echo "  cd ${CONDOR_DIR}"
        echo "  condor_submit_dag hybridnew.dag"
    else
        echo "Submitting DAG workflow..."
        cd "${CONDOR_DIR}"
        condor_submit_dag hybridnew.dag
        echo ""
        echo "Monitor with: condor_q -dag"
    fi
    echo "=========================================="
    exit 0
fi

# Step 2: Generate toys (local execution only - condor uses DAG)
if [[ "$MERGE_ONLY" == false && "$EXTRACT_ONLY" == false ]]; then
    echo ""
    echo "Generating toys..."

    for r in "${R_VALUES[@]}"; do
        echo "  Processing r = $r..."
        for ((j=0; j<NJOBS; j++)); do
            seed=$((1000 + j))
            TOY_CMD="combine -M HybridNew workspace.root \
                --LHCmode LHC-limits \
                --singlePoint $r \
                --saveToys \
                --saveHybridResult \
                -T ${NTOYS} \
                -s ${seed} \
                -n .r${r}.seed${seed}"

            run_cmd "$TOY_CMD"

            # Move output
            if [[ "$DRY_RUN" == false ]]; then
                mv -f higgsCombine.*.HybridNew.*.root "${TOYS_DIR}/" 2>/dev/null || true
            fi
        done
    done
fi

# Step 3: Merge results (skip if extract-only)
if [[ "$EXTRACT_ONLY" == false ]]; then
    echo ""
    echo "Merging toy results..."

    GRID_FILE="${OUTPUT_DIR}/hybridnew_grid.root"

    # Find all toy files
    TOY_FILES=$(ls "${TOYS_DIR}"/higgsCombine.*.root 2>/dev/null || true)
    if [[ -z "$TOY_FILES" ]]; then
        echo "ERROR: No toy files found in ${TOYS_DIR}/"
        exit 1
    fi

    run_cmd "hadd -f ${GRID_FILE} ${TOYS_DIR}/higgsCombine.*.root"
fi

# Step 4: Extract limits
echo ""
echo "Extracting limits..."

GRID_FILE="${OUTPUT_DIR}/hybridnew_grid.root"
if [[ ! -f "$GRID_FILE" ]]; then
    echo "ERROR: Grid file not found: $GRID_FILE"
    exit 1
fi

# Expected limits for each quantile
for q in "${QUANTILES[@]}"; do
    echo "  Extracting expected limit at quantile $q..."
    run_cmd "combine -M HybridNew workspace.root \
        --LHCmode LHC-limits \
        --readHybridResults \
        --grid=${GRID_FILE} \
        --expectedFromGrid $q \
        -n .${MASSPOINT}.${METHOD}.${BINNING}.exp${q} \
        -m 120 \
        2>&1 | tee ${OUTPUT_DIR}/combine_exp${q}.out"

    if [[ "$DRY_RUN" == false ]]; then
        mv -f higgsCombine.*.HybridNew.*.root "${OUTPUT_DIR}/" 2>/dev/null || true
    fi
done

# Observed limit
echo "  Extracting observed limit..."
run_cmd "combine -M HybridNew workspace.root \
    --LHCmode LHC-limits \
    --readHybridResults \
    --grid=${GRID_FILE} \
    -n .${MASSPOINT}.${METHOD}.${BINNING}.obs \
    -m 120 \
    2>&1 | tee ${OUTPUT_DIR}/combine_obs.out"

if [[ "$DRY_RUN" == false ]]; then
    mv -f higgsCombine.*.HybridNew.*.root "${OUTPUT_DIR}/" 2>/dev/null || true
fi

echo ""
echo "SUCCESS: HybridNew limits saved to ${OUTPUT_DIR}/"
echo "Done."
