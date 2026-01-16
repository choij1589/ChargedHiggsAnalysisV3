#!/bin/bash
set -euo pipefail

# Get the script directory (SignalRegionStudyV2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Mass points
MASSPOINTs_BASELINE=(
    "MHc70_MA15" "MHc70_MA18" "MHc70_MA40" "MHc70_MA55" "MHc70_MA65"
    "MHc85_MA15" "MHc85_MA70" "MHc85_MA80"      # 85_21 is missng in 16a
    "MHc100_MA15" "MHc100_MA24" "MHc100_MA60" "MHc100_MA75" "MHc100_MA95"
    "MHc115_MA15" "MHc115_MA27" "MHc115_MA87" "MHc115_MA110"
    "MHc130_MA15" "MHc130_MA30" "MHc130_MA55" "MHc130_MA83" "MHc130_MA90" "MHc130_MA100" "MHc130_MA125"
    "MHc145_MA15" "MHc145_MA35" "MHc145_MA92" "MHc145_MA140"
    "MHc160_MA15" "MHc160_MA50" "MHc160_MA85" "MHc160_MA98" "MHc160_MA120" "MHc160_MA135" "MHc160_MA155"
)
MASSPOINTs_PARTICLENET=(
    "MHc100_MA95" "MHc130_MA90" "MHc160_MA85" "MHc115_MA87" "MHc145_MA92" "MHc160_MA98"
)

# Eras
ERAs_RUN2=("2016preVFP" "2016postVFP" "2017" "2018")
ERAs_RUN3=("2022" "2022EE" "2023" "2023BPix")

# Channels - SR1E2Mu must be processed first (SR3Mu loads fit from SR1E2Mu)
CHANNELs=("SR1E2Mu" "SR3Mu")

# Number of parallel jobs
NJOBS_RUN2=8
NJOBS_RUN3=6
# Number of eras to process in parallel (set to 1 for sequential)
NJOBS_ERA=4

# Export PATH to include python scripts
export PATH="${PWD}/python:${PATH}"

# Parse command line arguments
MODE="all"  # Options: all, run2, run3
SINGLE_ERA=""  # Single era mode (overrides MODE)
METHOD="Baseline"  # Options: Baseline, ParticleNet
EXTRA_ARGS=""
BINNING="extended"
VALIDATION_ONLY=false
# DO_PLOT_SCORE is set after METHOD is parsed (default depends on method)
DO_PLOT_SCORE_SET=false
# Datacard and limits options (enabled by default)
DO_PRINT_DATACARD=true
DO_COMBINE_DATACARDS=true
DO_RUN_ASYMPTOTIC=true
# Dry run mode
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="${2,,}"  # Convert to lowercase
            shift 2
            ;;
        --era)
            SINGLE_ERA="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --binning)
            BINNING="$2"
            EXTRA_ARGS="$EXTRA_ARGS --binning $2"
            shift 2
            ;;
        --unblind)
            EXTRA_ARGS="$EXTRA_ARGS --unblind"
            shift
            ;;
        --partial-unblind)
            EXTRA_ARGS="$EXTRA_ARGS --partial-unblind"
            shift
            ;;
        --debug)
            EXTRA_ARGS="$EXTRA_ARGS --debug"
            shift
            ;;
        --njobs-era)
            NJOBS_ERA="$2"
            shift 2
            ;;
        --plot-score)
            DO_PLOT_SCORE_SET=true
            DO_PLOT_SCORE=true
            shift
            ;;
        --no-plot-score)
            DO_PLOT_SCORE_SET=true
            DO_PLOT_SCORE=false
            shift
            ;;
        --validationOnly)
            VALIDATION_ONLY=true
            shift
            ;;
        --printDatacard)
            DO_PRINT_DATACARD=true
            shift
            ;;
        --no-printDatacard)
            DO_PRINT_DATACARD=false
            shift
            ;;
        --combineDatacards)
            DO_COMBINE_DATACARDS=true
            shift
            ;;
        --no-combineDatacards)
            DO_COMBINE_DATACARDS=false
            shift
            ;;
        --runAsymptotic)
            DO_RUN_ASYMPTOTIC=true
            shift
            ;;
        --no-runAsymptotic)
            DO_RUN_ASYMPTOTIC=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--mode <all|run2|run3>] [--method <Baseline|ParticleNet>] [--binning <extended|uniform>] [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  all    - Process Run2 and Run3"
            echo "  run2   - Process Run2 only"
            echo "  run3   - Process Run3 only"
            echo ""
            echo "Single Era:"
            echo "  --era <era>  - Process single era only (e.g., --era 2018)"
            echo "                 Valid: 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix"
            echo ""
            echo "Methods:"
            echo "  Baseline    - Standard cut-based analysis"
            echo "  ParticleNet - MVA-based analysis (score plotting enabled by default)"
            echo ""
            echo "Template Options:"
            echo "  --binning extended   - Use extended binning (default, 19 bins)"
            echo "  --binning uniform    - Use uniform binning (15 bins)"
            echo "  --unblind            - Use real data for data_obs"
            echo "  --partial-unblind    - Unblind low score region (requires ParticleNet)"
            echo "  --validationOnly     - Skip template generation, run validation and plotting only"
            echo "  --plot-score         - Enable ParticleNet score plotting (default for ParticleNet)"
            echo "  --no-plot-score      - Disable ParticleNet score plotting"
            echo "  --njobs-era N        - Number of eras to process in parallel (default: 4)"
            echo "  --debug              - Enable debug logging"
            echo ""
            echo "Datacard & Limits Options (enabled by default):"
            echo "  --no-printDatacard      - Skip datacard generation"
            echo "  --no-combineDatacards   - Skip datacard combination"
            echo "  --no-runAsymptotic      - Skip asymptotic limits"
            echo ""
            echo "Other Options:"
            echo "  --dry-run               - Print commands without executing"
            echo ""
            echo "Examples:"
            echo "  # Full pipeline (default): templates + datacards + combination + limits"
            echo "  $0 --mode run2 --method Baseline"
            echo ""
            echo "  # Templates only (skip datacards and limits)"
            echo "  $0 --mode run2 --method Baseline --no-printDatacard --no-combineDatacards --no-runAsymptotic"
            echo ""
            echo "  # Datacards + limits only (on existing templates)"
            echo "  $0 --mode run2 --method Baseline --validationOnly"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set defaults: plot-score enabled for ParticleNet only (if not explicitly set)
if [[ "$DO_PLOT_SCORE_SET" == "false" ]]; then
    if [[ "$METHOD" == "ParticleNet" ]]; then
        DO_PLOT_SCORE=true
    else
        DO_PLOT_SCORE=false
    fi
fi

# Select mass points based on method
if [[ "$METHOD" == "ParticleNet" ]]; then
    MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}")
else
    MASSPOINTs=("${MASSPOINTs_BASELINE[@]}")
fi

echo "============================================================"
echo "SignalRegionStudyV2 Template Generation"
if [[ -n "$SINGLE_ERA" ]]; then
    echo "Era: $SINGLE_ERA (single era mode)"
else
    echo "Mode: $MODE"
fi
echo "Method: $METHOD"
echo "Mass points: ${MASSPOINTs[*]}"
echo "Binning: $BINNING"
echo "Validation only: $VALIDATION_ONLY"
echo "Plot score: $DO_PLOT_SCORE"
echo "Print datacard: $DO_PRINT_DATACARD"
echo "Combine datacards: $DO_COMBINE_DATACARDS"
echo "Run asymptotic: $DO_RUN_ASYMPTOTIC"
echo "Dry run: $DRY_RUN"
echo "Extra args: $EXTRA_ARGS"
echo "============================================================"

# =============================================================================
# Helper Functions
# =============================================================================

# Function to get the template directory path (absolute)
function get_template_dir() {
    local era=$1 channel=$2 masspoint=$3 method=$4 binning=$5 extra_args=${6:-}
    local binning_suffix="$binning"
    if [[ "$extra_args" == *"--partial-unblind"* ]]; then
        binning_suffix="${binning}_partial_unblind"
    elif [[ "$extra_args" == *"--unblind"* ]]; then
        binning_suffix="${binning}_unblind"
    fi
    echo "${SCRIPT_DIR}/templates/${era}/${channel}/${masspoint}/${method}/${binning_suffix}"
}
export -f get_template_dir

# Function to generate templates for a single configuration
function generate_template() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local method=$4
    local binning=$5
    local extra_args=${6:-}

    local cmd="python3 python/makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method $method $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"

    echo "Generating template: era=$era, channel=$channel, masspoint=$masspoint, method=$method"
    # Run python script and capture output (logs dir created AFTER since python script may recreate template dir)
    local output
    output=$(python3 python/makeBinnedTemplates.py --era "$era" --channel "$channel" --masspoint "$masspoint" --method "$method" $extra_args 2>&1)
    local exit_code=$?

    # Create logs directory after python script (which may have recreated the template dir)
    mkdir -p "$log_dir"
    echo "$output" > "${log_dir}/makeBinnedTemplates.log"
    echo "$output"

    return $exit_code
}
export -f generate_template

# Function to validate templates for a single configuration
function validate_template() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local method=$4
    local binning=$5
    local extra_args=${6:-}

    local cmd="python3 python/checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method $method --binning $binning $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Validating template: era=$era, channel=$channel, masspoint=$masspoint"
    python3 python/checkTemplates.py --era "$era" --channel "$channel" --masspoint "$masspoint" --method "$method" --binning "$binning" $extra_args \
        2>&1 | tee "${log_dir}/checkTemplates.log"
}
export -f validate_template

# Function to plot ParticleNet scores for a single configuration
function plot_score() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local binning=$4
    local extra_args=${5:-}

    local cmd="python3 python/plotParticleNetScore.py --era $era --channel $channel --masspoint $masspoint --binning $binning $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$METHOD" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Plotting ParticleNet score: era=$era, channel=$channel, masspoint=$masspoint"
    python3 python/plotParticleNetScore.py --era "$era" --channel "$channel" --masspoint "$masspoint" --binning "$binning" $extra_args \
        2>&1 | tee "${log_dir}/plotParticleNetScore.log"
}
export -f plot_score

# Function to generate datacards for a single configuration
function print_datacard() {
    local era=$1 channel=$2 masspoint=$3 method=$4 binning=$5 extra_args=${6:-}

    local cmd="python3 python/printDatacard.py --era $era --channel $channel --masspoint $masspoint --method $method --binning $binning $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Generating datacard: era=$era, channel=$channel, masspoint=$masspoint"
    python3 python/printDatacard.py --era "$era" --channel "$channel" \
        --masspoint "$masspoint" --method "$method" --binning "$binning" $extra_args \
        2>&1 | tee "${log_dir}/printDatacard.log"
}
export -f print_datacard

# Function to run asymptotic limits for a single configuration
function run_asymptotic() {
    local era=$1 channel=$2 masspoint=$3 method=$4 binning=$5

    local cmd="bash scripts/runAsymptotic.sh --era $era --channel $channel --masspoint $masspoint --method $method --binning $binning"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$method" "$binning" "")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Running asymptotic: era=$era, channel=$channel, masspoint=$masspoint"
    bash scripts/runAsymptotic.sh --era "$era" --channel "$channel" \
        --masspoint "$masspoint" --method "$method" --binning "$binning" \
        2>&1 | tee "${log_dir}/runAsymptotic.log"
}
export -f run_asymptotic

# Function to combine channels (SR1E2Mu + SR3Mu -> Combined)
function combine_channels() {
    local era=$1 masspoint=$2 method=$3 binning=$4 extra_args=${5:-}

    local cmd="python3 python/combineDatacards.py --mode channel --era $era --masspoint $masspoint --method $method --binning $binning $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "Combined" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Combining channels: era=$era, masspoint=$masspoint"
    python3 python/combineDatacards.py --mode channel \
        --era "$era" --masspoint "$masspoint" --method "$method" --binning "$binning" $extra_args \
        2>&1 | tee "${log_dir}/combineDatacards_channels.log"
}
export -f combine_channels

# Function to combine eras
function combine_eras() {
    local eras=$1 channel=$2 masspoint=$3 method=$4 binning=$5 output_era=$6 extra_args=${7:-}

    local cmd="python3 python/combineDatacards.py --mode era --eras $eras --channel $channel --masspoint $masspoint --method $method --binning $binning --output-era $output_era $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$output_era" "$channel" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Combining eras: eras=$eras -> $output_era, channel=$channel, masspoint=$masspoint"
    python3 python/combineDatacards.py --mode era \
        --eras "$eras" --channel "$channel" --masspoint "$masspoint" \
        --method "$method" --binning "$binning" --output-era "$output_era" $extra_args \
        2>&1 | tee "${log_dir}/combineDatacards_eras.log"
}
export -f combine_eras

# Determine extra args for various steps
VALIDATE_EXTRA_ARGS=""
DATACARD_EXTRA_ARGS=""
COMBINE_EXTRA_ARGS=""
PLOT_EXTRA_ARGS=""
if [[ "$EXTRA_ARGS" == *"--unblind"* ]]; then
    VALIDATE_EXTRA_ARGS="--unblind"
    DATACARD_EXTRA_ARGS="--unblind"
    PLOT_EXTRA_ARGS="--unblind"
elif [[ "$EXTRA_ARGS" == *"--partial-unblind"* ]]; then
    VALIDATE_EXTRA_ARGS="--partial-unblind"
    DATACARD_EXTRA_ARGS="--partial-unblind"
    COMBINE_EXTRA_ARGS="--partial-unblind"
    PLOT_EXTRA_ARGS="--partial-unblind"
fi

# =============================================================================
# Per-Era Processing Function
# =============================================================================
# Processes a single era through the full pipeline:
# templates -> validation -> datacards -> channel combination -> asymptotic

function process_era() {
    local era=$1
    local njobs=$2

    # Reconstruct arrays from exported strings (bash arrays can't be exported directly)
    local -a MASSPOINTs=($MASSPOINTs_STR)
    local -a CHANNELs=($CHANNELs_STR)

    echo ""
    echo "============================================================"
    echo "Processing era: $era (njobs=$njobs)"
    echo "============================================================"

    # Step 1: Template Generation (skip if --validationOnly)
    if [[ "$VALIDATION_ONLY" == "false" ]]; then
        echo "--- Generating templates for $era ---"
        # SR1E2Mu first (required for SR3Mu fit) - parallelize across mass points
        parallel -j "$njobs" generate_template "$era" "SR1E2Mu" {1} "$METHOD" "$BINNING" "\"$EXTRA_ARGS\"" \
            ::: "${MASSPOINTs[@]}"
        # Then SR3Mu - parallelize across mass points
        parallel -j "$njobs" generate_template "$era" "SR3Mu" {1} "$METHOD" "$BINNING" "\"$EXTRA_ARGS\"" \
            ::: "${MASSPOINTs[@]}"
    fi

    # Step 2: Validation - parallelize across channels and mass points
    echo "--- Validating templates for $era ---"
    parallel -j "$njobs" validate_template "$era" {1} {2} "$METHOD" "$BINNING" "\"$VALIDATE_EXTRA_ARGS\"" \
        ::: "${CHANNELs[@]}" ::: "${MASSPOINTs[@]}"

    # Step 3: ParticleNet Score Plotting (if enabled)
    if [[ "$DO_PLOT_SCORE" == "true" && "$METHOD" == "ParticleNet" ]]; then
        echo "--- Plotting ParticleNet scores for $era ---"
        parallel -j "$njobs" plot_score "$era" {1} {2} "$BINNING" "\"$PLOT_EXTRA_ARGS\"" \
            ::: "${CHANNELs[@]}" ::: "${MASSPOINTs[@]}"
    fi

    # Step 4: Datacard Generation - parallelize across channels and mass points
    if [[ "$DO_PRINT_DATACARD" == "true" ]]; then
        echo "--- Generating datacards for $era ---"
        parallel -j "$njobs" print_datacard "$era" {1} {2} "$METHOD" "$BINNING" "\"$DATACARD_EXTRA_ARGS\"" \
            ::: "${CHANNELs[@]}" ::: "${MASSPOINTs[@]}"
    fi

    # Step 5: Combine Channels (SR1E2Mu + SR3Mu -> Combined) - parallelize across mass points
    if [[ "$DO_COMBINE_DATACARDS" == "true" ]]; then
        echo "--- Combining channels for $era ---"
        parallel -j "$njobs" combine_channels "$era" {1} "$METHOD" "$BINNING" "\"$COMBINE_EXTRA_ARGS\"" \
            ::: "${MASSPOINTs[@]}"
    fi

    # Step 6: Asymptotic Limits for this era - parallelize across mass points
    if [[ "$DO_RUN_ASYMPTOTIC" == "true" ]]; then
        echo "--- Running asymptotic limits for $era ---"
        parallel -j "$njobs" run_asymptotic "$era" "Combined" {1} "$METHOD" "$BINNING" \
            ::: "${MASSPOINTs[@]}"
    fi

    echo "Era $era processing complete!"
}

# Export function and variables for parallel
# NOTE: Bash arrays cannot be exported directly. Convert to strings for parallel.
export MASSPOINTs_STR="${MASSPOINTs[*]}"
export CHANNELs_STR="${CHANNELs[*]}"
export -f process_era
export -f get_template_dir generate_template validate_template plot_score print_datacard run_asymptotic
export -f combine_channels combine_eras
export SCRIPT_DIR METHOD BINNING EXTRA_ARGS VALIDATION_ONLY DO_PLOT_SCORE DO_PRINT_DATACARD
export DO_COMBINE_DATACARDS DO_RUN_ASYMPTOTIC DRY_RUN
export VALIDATE_EXTRA_ARGS DATACARD_EXTRA_ARGS COMBINE_EXTRA_ARGS PLOT_EXTRA_ARGS

# =============================================================================
# Main Execution
# =============================================================================

# Single era mode
if [[ -n "$SINGLE_ERA" ]]; then
    # Determine njobs based on era
    if [[ " ${ERAs_RUN2[*]} " =~ " ${SINGLE_ERA} " ]]; then
        process_era "$SINGLE_ERA" $NJOBS_RUN2
    elif [[ " ${ERAs_RUN3[*]} " =~ " ${SINGLE_ERA} " ]]; then
        process_era "$SINGLE_ERA" $NJOBS_RUN3
    else
        echo "ERROR: Unknown era '$SINGLE_ERA'"
        echo "Valid eras: ${ERAs_RUN2[*]} ${ERAs_RUN3[*]}"
        exit 1
    fi
    echo ""
    echo "============================================================"
    echo "Single era processing complete!"
    echo "============================================================"
    exit 0
fi

# Multi-era mode: Process eras in parallel
if [[ "$MODE" == "all" || "$MODE" == "run2" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run2 eras (parallel=$NJOBS_ERA)..."
    echo "============================================================"
    parallel -j $NJOBS_ERA process_era {1} $NJOBS_RUN2 ::: "${ERAs_RUN2[@]}"
fi

if [[ "$MODE" == "all" || "$MODE" == "run3" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run3 eras (parallel=$NJOBS_ERA)..."
    echo "============================================================"
    parallel -j $NJOBS_ERA process_era {1} $NJOBS_RUN3 ::: "${ERAs_RUN3[@]}"
fi

# =============================================================================
# Combine Eras and Run Final Asymptotic
# =============================================================================

if [[ "$DO_COMBINE_DATACARDS" == "true" ]]; then
    echo ""
    echo "============================================================"
    echo "Combining eras..."
    echo "============================================================"

    # Run2 combination - parallelize across mass points
    if [[ "$MODE" == "all" || "$MODE" == "run2" ]]; then
        echo "Combining Run2 eras..."
        parallel -j $NJOBS_RUN2 combine_eras "2016preVFP,2016postVFP,2017,2018" "Combined" {1} \
            "$METHOD" "$BINNING" "Run2" "\"$COMBINE_EXTRA_ARGS\"" \
            ::: "${MASSPOINTs[@]}"
    fi
    # Run3 combination - parallelize across mass points
    if [[ "$MODE" == "all" || "$MODE" == "run3" ]]; then
        echo "Combining Run3 eras..."
        parallel -j $NJOBS_RUN3 combine_eras "2022,2022EE,2023,2023BPix" "Combined" {1} \
            "$METHOD" "$BINNING" "Run3" "\"$COMBINE_EXTRA_ARGS\"" \
            ::: "${MASSPOINTs[@]}"
    fi
fi

# Asymptotic on Run2/Run3 combined
if [[ "$DO_RUN_ASYMPTOTIC" == "true" ]]; then
    echo ""
    echo "============================================================"
    echo "Running asymptotic on combined eras..."
    echo "============================================================"

    if [[ "$MODE" == "all" || "$MODE" == "run2" ]]; then
        echo "Running asymptotic for Run2/Combined..."
        parallel -j $NJOBS_RUN2 run_asymptotic "Run2" "Combined" {1} "$METHOD" "$BINNING" \
            ::: "${MASSPOINTs[@]}"
    fi
    if [[ "$MODE" == "all" || "$MODE" == "run3" ]]; then
        echo "Running asymptotic for Run3/Combined..."
        parallel -j $NJOBS_RUN3 run_asymptotic "Run3" "Combined" {1} "$METHOD" "$BINNING" \
            ::: "${MASSPOINTs[@]}"
    fi
fi

# =============================================================================
# Full Combination (Run2 + Run3 -> All)
# =============================================================================

if [[ "$MODE" == "all" ]]; then
    if [[ "$DO_COMBINE_DATACARDS" == "true" ]]; then
        echo ""
        echo "============================================================"
        echo "Combining Run2 + Run3 -> All..."
        echo "============================================================"

        parallel -j $NJOBS_RUN2 combine_eras "Run2,Run3" "Combined" {1} \
            "$METHOD" "$BINNING" "All" "\"$COMBINE_EXTRA_ARGS\"" \
            ::: "${MASSPOINTs[@]}"
    fi

    if [[ "$DO_RUN_ASYMPTOTIC" == "true" ]]; then
        echo ""
        echo "============================================================"
        echo "Running asymptotic on All/Combined..."
        echo "============================================================"

        parallel -j $NJOBS_RUN2 run_asymptotic "All" "Combined" {1} "$METHOD" "$BINNING" \
            ::: "${MASSPOINTs[@]}"
    fi
fi

echo ""
echo "============================================================"
echo "All processing complete!"
echo "============================================================"
