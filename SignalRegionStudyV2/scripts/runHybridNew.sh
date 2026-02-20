#!/bin/bash
#
# runHybridNew.sh - Run HybridNew limit calculation with toys via HTCondor
#
# This script supports Condor-only execution for SignalRegionStudyV2.
# For Run2+Run3 combined limit extraction.
#
# Usage:
#   ./runHybridNew.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended --auto-grid
#   ./runHybridNew.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended --dry-run
#

set -e

# Default values
ERA=""
CHANNEL="Combined"
MASSPOINT=""
METHOD="Baseline"
BINNING="extended"
PARTIAL_UNBLIND=false
NTOYS=100
NJOBS=10
RMIN=0.0
RMAX=2.0
RSTEP=0.2
DRY_RUN=false
VERBOSE=false
MERGE_ONLY=false
EXTRACT_ONLY=false
AUTO_GRID=false
PARTIAL_EXTRACT=false
PLOT_ONLY=false

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
        --partial-unblind)
            PARTIAL_UNBLIND=true
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
        --partial-extract)
            PARTIAL_EXTRACT=true
            shift
            ;;
        --plot)
            PLOT_ONLY=true
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
            echo "  --era        Data-taking period (Run2, Run3, All, or individual era)"
            echo "  --channel    Analysis channel (default: Combined)"
            echo "  --masspoint  Signal mass point (e.g., MHc130_MA90)"
            echo ""
            echo "Options:"
            echo "  --method     Template method (Baseline, ParticleNet) [default: Baseline]"
            echo "  --binning    Binning scheme (uniform, extended) [default: extended]"
            echo "  --partial-unblind  Use partial-unblind templates"
            echo "  --ntoys      Number of toys per job [default: 100]"
            echo "  --njobs      Number of parallel jobs per r-point [default: 10]"
            echo "  --rmin       Minimum r value [default: 0.0]"
            echo "  --rmax       Maximum r value [default: 2.0]"
            echo "  --rstep      r value step size [default: 0.2]"
            echo "  --auto-grid  Auto-tune grid from Asymptotic results (overrides rmin/rmax/rstep)"
            echo ""
            echo "Post-processing Options:"
            echo "  --merge-only     Only merge existing toy results"
            echo "  --extract-only   Only extract limits from merged grid"
            echo "  --partial-extract  Extract limits from incomplete condor jobs"
            echo "  --plot           Generate plots from existing HybridNew output"
            echo ""
            echo "Other Options:"
            echo "  --dry-run    Prepare DAG files without submitting"
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
if [[ -z "$ERA" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era and --masspoint are required"
    exit 1
fi

# Get WORKDIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Template directory with partial-unblind suffix handling
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

# Create output directories
OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/hybridnew"
CONDOR_DIR="${OUTPUT_DIR}/condor"

# For fresh submission, clean up existing directory
if [[ "$MERGE_ONLY" == false && "$EXTRACT_ONLY" == false && "$PARTIAL_EXTRACT" == false && "$PLOT_ONLY" == false ]]; then
    if [[ -d "$OUTPUT_DIR" ]]; then
        echo "Cleaning up existing output directory: $OUTPUT_DIR"
        rm -rf "$OUTPUT_DIR"
    fi
fi
mkdir -p "$OUTPUT_DIR" "$CONDOR_DIR"

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

# ============================================================
# Plotting Function
# ============================================================
generate_hybridnew_plots() {
    local INPUT_DIR="$1"
    local PLOTS_DIR="${INPUT_DIR}/plots"

    echo ""
    echo "Generating plots..."
    mkdir -p "${PLOTS_DIR}"

    # Check if required files exist - create grid if needed
    if [[ ! -f "${INPUT_DIR}/hybridnew_grid.root" ]]; then
        local TOY_COUNT=$(ls "${INPUT_DIR}"/higgsCombine.r*.root 2>/dev/null | wc -l)
        if [[ "$TOY_COUNT" -gt 0 ]]; then
            echo "  Merging ${TOY_COUNT} toy files..."
            hadd -f "${INPUT_DIR}/hybridnew_grid.root" "${INPUT_DIR}"/higgsCombine.r*.root 2>&1 | tail -1
        else
            echo "ERROR: No toy files found in ${INPUT_DIR}"
            return 1
        fi
    fi

    # Plot CLs vs r curve (pass directory to read individual toy files, filtering zero CLs)
    echo "  Creating CLs vs r plot..."
    python3 ${WORKDIR}/SignalRegionStudyV2/python/plotHybridNewGrid.py \
        "${INPUT_DIR}" \
        -o "${PLOTS_DIR}/cls_vs_r.png" \
        --title "${MASSPOINT} ${METHOD} ${BINNING_SUFFIX}"

    # Plot test statistic distributions for all r values
    echo "  Creating test statistic distribution plots..."
    python3 << EOFPYTHON
import os
import subprocess
from collections import defaultdict

input_dir = "${INPUT_DIR}"
plots_dir = "${PLOTS_DIR}"
workdir = "${WORKDIR}"
masspoint = "${MASSPOINT}"
method = "${METHOD}"

# Find all unique r values from toy files
completed = defaultdict(int)
for fname in os.listdir(input_dir):
    if fname.startswith("higgsCombine.r") and fname.endswith(".root"):
        if "partial" in fname or "exp" in fname or "obs" in fname:
            continue
        try:
            r_str = fname.split(".r")[1].split(".seed")[0]
            r = float(r_str)
            completed[r] += 1
        except:
            pass

print(f"    Found {len(completed)} r-value points")

# Plot test statistic for each r value with at least 2 toys
for r in sorted(completed.keys()):
    if completed[r] >= 2:
        output_file = os.path.join(plots_dir, f"test_stat_r{r:.4f}.png")
        script_path = os.path.join(workdir, "SignalRegionStudyV2/python/plotTestStatDist.py")
        grid_file = os.path.join(input_dir, "hybridnew_grid.root")

        cmd = [
            "python3", script_path, grid_file,
            "--r", str(r),
            "-o", output_file,
            "--title", f"{masspoint} r={r:.2f}"
        ]
        try:
            subprocess.run(cmd, capture_output=True, check=True)
            print(f"    r={r:.4f}: OK")
        except subprocess.CalledProcessError:
            print(f"    r={r:.4f}: FAILED")
EOFPYTHON

    echo "Plots saved to: ${PLOTS_DIR}/"
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
    # Use a temp file to capture output and handle errors properly
    AUTO_GRID_OUTPUT=$(python3 -c "
import sys
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kError  # Suppress ROOT warnings

    f = ROOT.TFile.Open('${ASYMPTOTIC_FILE}')
    if not f or f.IsZombie():
        print('0.01 2.0 0.1', file=sys.stdout)
        sys.exit(0)

    tree = f.Get('limit')
    if not tree:
        print('0.01 2.0 0.1', file=sys.stdout)
        f.Close()
        sys.exit(0)

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

        print(f'{rmin:.4f} {rmax:.4f} {rstep:.4f}', file=sys.stdout)
    else:
        print('0.01 2.0 0.1', file=sys.stdout)  # fallback
except Exception as e:
    print(f'0.01 2.0 0.1', file=sys.stdout)  # fallback on any error
    sys.exit(0)
" 2>/dev/null)

    # Parse the output
    if [[ -n "$AUTO_GRID_OUTPUT" ]]; then
        read RMIN RMAX RSTEP <<< "$AUTO_GRID_OUTPUT"
    else
        echo "  WARNING: Could not read asymptotic results, using defaults"
        RMIN=0.01
        RMAX=2.0
        RSTEP=0.1
    fi

    echo "  Auto-tuned: rmin=${RMIN}, rmax=${RMAX}, rstep=${RSTEP}"
fi

# ============================================================
# Plot Only Mode (skip if --partial-extract is also specified)
# ============================================================
if [[ "$PLOT_ONLY" == true && "$PARTIAL_EXTRACT" == false ]]; then
    echo ""
    echo "=== Generate HybridNew Plots ==="
    echo "Mass point: ${MASSPOINT}"
    echo "Method: ${METHOD}"
    echo "Binning: ${BINNING_SUFFIX}"

    # Find input directory (try partial_extract first, then condor)
    if [[ -d "${TEMPLATE_DIR}/combine_output/hybridnew/partial_extract" ]]; then
        INPUT_DIR="${TEMPLATE_DIR}/combine_output/hybridnew/partial_extract"
    elif [[ -d "${TEMPLATE_DIR}/combine_output/hybridnew/condor" ]]; then
        INPUT_DIR="${TEMPLATE_DIR}/combine_output/hybridnew/condor"
    else
        echo "ERROR: No HybridNew output directory found"
        exit 1
    fi

    echo "Input directory: ${INPUT_DIR}"
    generate_hybridnew_plots "${INPUT_DIR}"
    exit 0
fi

# ============================================================
# Partial Extract Mode
# ============================================================
if [[ "$PARTIAL_EXTRACT" == true ]]; then
    echo ""
    echo "=== Partial HybridNew Limit Extraction ==="
    echo "Mass point: ${MASSPOINT}"
    echo "Method: ${METHOD}"
    echo "Binning: ${BINNING_SUFFIX}"
    echo ""

    # Check condor directory exists
    CONDOR_TOYS_DIR="${TEMPLATE_DIR}/combine_output/hybridnew/condor"
    if [[ ! -d "$CONDOR_TOYS_DIR" ]]; then
        echo "ERROR: No condor directory found at $CONDOR_TOYS_DIR"
        exit 1
    fi

    # Check workspace exists
    if [[ ! -f "${CONDOR_TOYS_DIR}/workspace.root" ]]; then
        echo "ERROR: No workspace.root found in $CONDOR_TOYS_DIR"
        exit 1
    fi

    # Create partial_extract directory
    PARTIAL_DIR="${TEMPLATE_DIR}/combine_output/hybridnew/partial_extract"
    mkdir -p "${PARTIAL_DIR}"
    echo "Output directory: ${PARTIAL_DIR}"

    # Copy workspace.root
    cp "${CONDOR_TOYS_DIR}/workspace.root" "${PARTIAL_DIR}/"

    # Count and copy toy files
    TOY_FILES=$(ls "${CONDOR_TOYS_DIR}"/higgsCombine.r*.root 2>/dev/null | wc -l)
    if [[ "$TOY_FILES" -eq 0 ]]; then
        echo "ERROR: No toy files found in ${CONDOR_TOYS_DIR}/"
        exit 1
    fi

    echo "Copying ${TOY_FILES} completed toy files..."
    cp "${CONDOR_TOYS_DIR}"/higgsCombine.r*.root "${PARTIAL_DIR}/"

    # Report r-value coverage
    echo ""
    echo "R-value coverage:"
    python3 << EOFPYTHON
import os
from collections import defaultdict

partial_dir = "${PARTIAL_DIR}"
completed = defaultdict(int)

for fname in os.listdir(partial_dir):
    if fname.startswith("higgsCombine.r") and fname.endswith(".root") and "partial" not in fname:
        try:
            r_str = fname.split(".r")[1].split(".seed")[0]
            r = float(r_str)
            completed[r] += 1
        except:
            pass

total_toys = sum(completed.values())
complete_r = sorted([r for r, c in completed.items() if c >= 10])
partial_r = sorted([r for r, c in completed.items() if 0 < c < 10])

print(f"  Total toy files: {total_toys}")
print(f"  Complete r-points (>=10 toys): {len(complete_r)}")
if complete_r:
    print(f"    Range: r = {min(complete_r):.2f} - {max(complete_r):.2f}")
if partial_r:
    print(f"  Partial r-points (<10 toys): {len(partial_r)}")
    print(f"    Range: r = {min(partial_r):.2f} - {max(partial_r):.2f}")
    print(f"    WARNING: Partial r-points may affect limit accuracy")
EOFPYTHON

    # Merge toys
    echo ""
    echo "Merging toy files..."
    cd "${PARTIAL_DIR}"
    hadd -f hybridnew_grid.root higgsCombine.r*.root 2>&1 | tail -3

    # Extract limits for each quantile
    echo ""
    echo "Extracting limits..."
    for q in 0.025 0.160 0.500 0.840 0.975; do
        echo "  Quantile $q..."
        combine -M HybridNew workspace.root \
            --LHCmode LHC-limits \
            --readHybridResults \
            --grid=hybridnew_grid.root \
            --expectedFromGrid $q \
            -n .partial.exp${q} \
            -m 120 \
            2>&1 | grep -E "^(Limit|Expected)" || true
    done

    # Extract observed limit
    echo "  Observed..."
    combine -M HybridNew workspace.root \
        --LHCmode LHC-limits \
        --readHybridResults \
        --grid=hybridnew_grid.root \
        -n .partial.obs \
        -m 120 \
        2>&1 | grep -E "^(Limit|Observed)" || true

    # Print summary
    echo ""
    echo "=== Extracted Limits ==="
    python3 << EOFPYTHON
import ROOT
import os

partial_dir = "${PARTIAL_DIR}"
ROOT.gROOT.SetBatch(True)

quantiles = {
    "exp-2": "partial.exp0.025",
    "exp-1": "partial.exp0.160",
    "median": "partial.exp0.500",
    "exp+1": "partial.exp0.840",
    "exp+2": "partial.exp0.975",
    "obs": "partial.obs"
}

results = {}
for label, suffix in quantiles.items():
    # Find file
    for fname in os.listdir(partial_dir):
        if suffix in fname and fname.endswith(".root"):
            fpath = os.path.join(partial_dir, fname)
            f = ROOT.TFile.Open(fpath)
            if f and not f.IsZombie():
                tree = f.Get("limit")
                if tree and tree.GetEntries() > 0:
                    tree.GetEntry(0)
                    results[label] = tree.limit
                f.Close()
            break

# Print results
print(f"  -2σ expected: {results.get('exp-2', 'N/A'):.4f}" if 'exp-2' in results else "  -2σ expected: N/A")
print(f"  -1σ expected: {results.get('exp-1', 'N/A'):.4f}" if 'exp-1' in results else "  -1σ expected: N/A")
print(f"  Median expected: {results.get('median', 'N/A'):.4f}" if 'median' in results else "  Median expected: N/A")
print(f"  +1σ expected: {results.get('exp+1', 'N/A'):.4f}" if 'exp+1' in results else "  +1σ expected: N/A")
print(f"  +2σ expected: {results.get('exp+2', 'N/A'):.4f}" if 'exp+2' in results else "  +2σ expected: N/A")
print(f"  Observed: {results.get('obs', 'N/A'):.4f}" if 'obs' in results else "  Observed: N/A")
EOFPYTHON

    # Generate plots
    generate_hybridnew_plots "${PARTIAL_DIR}"

    echo ""
    echo "Results saved to: ${PARTIAL_DIR}/"
    echo "Done."
    exit 0
fi

# ============================================================
# Condor DAG Submission
# ============================================================

echo "HybridNew limit calculation for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})"
echo "  Toys per job: ${NTOYS}"
echo "  Number of jobs: ${NJOBS}"
echo "  r range: [${RMIN}, ${RMAX}] step ${RSTEP}"
echo ""

# Generate r values using Python
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

echo ""
echo "Preparing DAG workflow..."

# Create logs directory
mkdir -p "${CONDOR_DIR}/logs"

# CMSSW path for Combine (use Common directory)
CMSSW_BASE="${WORKDIR}/Common/CMSSW_14_1_0_pre4/src"

# Check if CMSSW exists
if [[ ! -d "$CMSSW_BASE" ]]; then
    echo "ERROR: CMSSW directory not found: $CMSSW_BASE"
    exit 1
fi

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
request_memory = 1GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = ${WORKSPACE_INPUT_FILES}
transfer_output_files = workspace.root
when_to_transfer_output = ON_EXIT

queue
EOF

# ========== toy.sh ==========
cat > "${CONDOR_DIR}/toy.sh" << EOFTOY
#!/bin/bash
R_VALUE=\$1
SEED=\$2

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

combine -M HybridNew workspace.root \\
    --LHCmode LHC-limits \\
    --singlePoint \${R_VALUE} \\
    --saveToys \\
    --saveHybridResult \\
    --expectSignal 0 \\
    -T ${NTOYS} \\
    -t -1 \\
    -s \${SEED} \\
    --cminDefaultMinimizerStrategy 0 \\
    --cminDefaultMinimizerTolerance 0.1 \\
    --cminFallbackAlgo Minuit2,Simplex,0:0.2 \\
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
request_memory = 2GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = workspace.root
transfer_output_files = higgsCombine.r\$(r_value).seed\$(seed).HybridNew.mH120.\$(seed).root
when_to_transfer_output = ON_EXIT

queue
EOF

# ========== merge.sh ==========
cat > "${CONDOR_DIR}/merge.sh" << EOFMERGE
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
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
request_memory = 2GB
request_disk = 2GB

should_transfer_files = YES
transfer_input_files = ${TOY_FILES_LIST}
transfer_output_files = hybridnew_grid.root
when_to_transfer_output = ON_EXIT

queue
EOF

# ========== extract.sh ==========
cat > "${CONDOR_DIR}/extract.sh" << EOFEXTRACT
#!/bin/bash
QUANTILE=\$1
NAME=\$2

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
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
request_memory = 1GB
request_disk = 500MB

should_transfer_files = YES
transfer_input_files = workspace.root,hybridnew_grid.root
transfer_output_files = combine_\$(name).out
when_to_transfer_output = ON_EXIT

queue
EOF

# ========== hybridnew.dag ==========
DAG_FILE="${CONDOR_DIR}/hybridnew.dag"

{
    echo "# HybridNew DAG workflow"
    echo "# Generated for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})"
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
