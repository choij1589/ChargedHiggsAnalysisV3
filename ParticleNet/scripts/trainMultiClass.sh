#!/bin/bash

# Multi-class training script for ParticleNet
# Trains 4-class models (signal vs 3 backgrounds) across all folds in parallel

set -e  # Exit on error

# Setup environment
export PATH="${PWD}/python:${PATH}"
export WORKDIR="${WORKDIR:-${PWD}/../..}"

# Check if we're in the right directory
if [[ ! -f "python/trainMultiClass.py" ]]; then
    echo "ERROR: Must run from ParticleNet directory"
    echo "Expected to find python/trainMultiClass.py"
    exit 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --channel CHANNEL     Channel to train (Run1E2Mu, Run3Mu) [required]"
    echo "  --signals SIGNALS     Comma-separated list of signal samples [optional]"
    echo "  --backgrounds BKGS    Space-separated list of background samples [required]"
    echo "  --model MODEL         Model type (ParticleNet, ParticleNetV2) [default: ParticleNet]"
    echo "  --loss-type LOSS      Loss function (weighted_ce, sample_normalized, focal) [default: weighted_ce]"
    echo "  --nNodes NODES        Hidden layer size [default: 128]"
    echo "  --initLR LR           Initial learning rate [default: 0.001]"
    echo "  --optimizer OPT       Optimizer (Adam, RMSprop, Adadelta) [default: Adam]"
    echo "  --scheduler SCHED     LR scheduler (StepLR, ExponentialLR, CyclicLR, ReduceLROnPlateau) [default: StepLR]"
    echo "  --max-epochs EPOCHS   Maximum training epochs [default: 81]"
    echo "  --pilot               Use pilot datasets for quick testing"
    echo "  --separate-bjets      Use separate b-jets as distinct particles"
    echo "  --debug               Enable debug logging"
    echo "  --dry-run             Show commands that would be executed"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Standard 3-background training"
    echo "  $0 --channel Run1E2Mu --backgrounds \"Skim_TriLep_TTLL_powheg Skim_TriLep_WZTo3LNu_amcatnlo Skim_TriLep_TTZToLLNuNu\""
    echo "  # 4-background training with tZq"
    echo "  $0 --channel Run3Mu --backgrounds \"Skim_TriLep_TTLL_powheg Skim_TriLep_WZTo3LNu_amcatnlo Skim_TriLep_TTZToLLNuNu Skim_TriLep_tZq\""
    echo "  # 2-background training"
    echo "  $0 --channel Run1E2Mu --backgrounds \"Skim_TriLep_TTLL_powheg Skim_TriLep_TTZToLLNuNu\" --pilot"
}

# Default parameters
CHANNEL=""
SIGNALS=""
BACKGROUNDS=""
MODEL="ParticleNet"
LOSS_TYPE="weighted_ce"
NNODES=128
INIT_LR=0.001
OPTIMIZER="Adam"
SCHEDULER="StepLR"
MAX_EPOCHS=81
PILOT=false
SEPARATE_BJETS=false
DEBUG=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --channel)
            CHANNEL="$2"
            shift 2
            ;;
        --signals)
            SIGNALS="$2"
            shift 2
            ;;
        --backgrounds)
            BACKGROUNDS="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --loss-type)
            LOSS_TYPE="$2"
            shift 2
            ;;
        --nNodes)
            NNODES="$2"
            shift 2
            ;;
        --initLR)
            INIT_LR="$2"
            shift 2
            ;;
        --optimizer)
            OPTIMIZER="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --pilot)
            PILOT=true
            shift
            ;;
        --separate-bjets)
            SEPARATE_BJETS=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CHANNEL" ]]; then
    echo "ERROR: --channel is required"
    usage
    exit 1
fi

if [[ -z "$BACKGROUNDS" ]]; then
    echo "ERROR: --backgrounds is required"
    usage
    exit 1
fi

# Validate channel
if [[ "$CHANNEL" != "Run1E2Mu" && "$CHANNEL" != "Run3Mu" && "$CHANNEL" != "Combined" ]]; then
    echo "ERROR: Invalid channel '$CHANNEL'. Must be Run1E2Mu, Run3Mu, or Combined"
    exit 1
fi

# Default signal samples if not specified
if [[ -z "$SIGNALS" ]]; then
    SIGNALS="TTToHcToWAToMuMu-MHc100MA95,TTToHcToWAToMuMu-MHc130MA90,TTToHcToWAToMuMu-MHc130MA100,TTToHcToWAToMuMu-MHc160MA85,TTToHcToWAToMuMu-MHc160MA155"
fi

# Convert comma-separated signals to array
IFS=',' read -ra SIGNAL_ARRAY <<< "$SIGNALS"

# Convert space-separated backgrounds to array (already space-separated from command line)
BACKGROUND_ARRAY=($BACKGROUNDS)

echo "=========================================="
echo "Multi-Class ParticleNet Training"
echo "=========================================="
echo "Channel: $CHANNEL"
echo "Signals: ${SIGNAL_ARRAY[*]}"
echo "Backgrounds: ${BACKGROUND_ARRAY[*]}"
echo "Number of classes: $((1 + ${#BACKGROUND_ARRAY[@]}))"
echo "Model: $MODEL"
echo "Loss type: $LOSS_TYPE"
echo "Hidden nodes: $NNODES"
echo "Learning rate: $INIT_LR"
echo "Optimizer: $OPTIMIZER"
echo "Scheduler: $SCHEDULER"
echo "Max epochs: $MAX_EPOCHS"
echo "Pilot mode: $PILOT"
echo "Separate b-jets: $SEPARATE_BJETS"
echo "Debug mode: $DEBUG"
echo "Dry run: $DRY_RUN"
echo "=========================================="

# Build base training command
BASE_CMD="python trainMultiClass.py"
BASE_CMD="$BASE_CMD --channel $CHANNEL"
BASE_CMD="$BASE_CMD --model $MODEL"
BASE_CMD="$BASE_CMD --loss_type $LOSS_TYPE"
BASE_CMD="$BASE_CMD --nNodes $NNODES"
BASE_CMD="$BASE_CMD --initLR $INIT_LR"
BASE_CMD="$BASE_CMD --optimizer $OPTIMIZER"
BASE_CMD="$BASE_CMD --scheduler $SCHEDULER"
BASE_CMD="$BASE_CMD --max_epochs $MAX_EPOCHS"
BASE_CMD="$BASE_CMD --weight_decay 1e-4"

if [[ "$PILOT" == true ]]; then
    BASE_CMD="$BASE_CMD --pilot"
fi

if [[ "$SEPARATE_BJETS" == true ]]; then
    BASE_CMD="$BASE_CMD --separate_bjets"
fi

if [[ "$DEBUG" == true ]]; then
    BASE_CMD="$BASE_CMD --debug"
fi

# Generate all training commands
COMMANDS=()
for signal in "${SIGNAL_ARRAY[@]}"; do
    for fold in {0..4}; do
        cmd="$BASE_CMD --signal $signal --fold $fold --backgrounds ${BACKGROUND_ARRAY[*]}"
        COMMANDS+=("$cmd")
    done
done

echo "Generated ${#COMMANDS[@]} training commands"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "DRY RUN - Commands that would be executed:"
    echo "=========================================="
    for i in "${!COMMANDS[@]}"; do
        printf "%3d: %s\n" $((i+1)) "${COMMANDS[$i]}"
    done
    echo ""
    echo "To execute, run without --dry-run"
    exit 0
fi

# Check if datasets exist before starting training
echo "Checking dataset availability..."
if [[ "$SEPARATE_BJETS" == true ]]; then
    DATASET_ROOT="$WORKDIR/ParticleNet/dataset_bjets/samples"
else
    DATASET_ROOT="$WORKDIR/ParticleNet/dataset/samples"
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "ERROR: Dataset root not found: $DATASET_ROOT"
    echo "Please run dataset creation first"
    exit 1
fi

# Check signal datasets
missing_signals=()
for signal in "${SIGNAL_ARRAY[@]}"; do
    signal_dir="$DATASET_ROOT/signals/$signal"
    if [[ ! -d "$signal_dir" ]]; then
        missing_signals+=("$signal")
    fi
done

# Check background datasets
missing_backgrounds=()
IFS=',' read -ra BG_ARRAY <<< "$BACKGROUND_SAMPLES"
for bg in "${BG_ARRAY[@]}"; do
    bg_dir="$DATASET_ROOT/backgrounds/$bg"
    if [[ ! -d "$bg_dir" ]]; then
        missing_backgrounds+=("$bg")
    fi
done

if [[ ${#missing_signals[@]} -gt 0 ]]; then
    echo "ERROR: Missing signal datasets:"
    printf "  %s\n" "${missing_signals[@]}"
    exit 1
fi

if [[ ${#missing_backgrounds[@]} -gt 0 ]]; then
    echo "ERROR: Missing background datasets:"
    printf "  %s\n" "${missing_backgrounds[@]}"
    exit 1
fi

echo "All required datasets found"
echo ""

# Execute training commands in parallel
echo "Starting parallel training..."
echo "Using GNU parallel with $(nproc) cores"
echo ""

# Create a temporary file with all commands
TEMP_CMD_FILE=$(mktemp)
printf "%s\n" "${COMMANDS[@]}" > "$TEMP_CMD_FILE"

# Run with GNU parallel
# --jobs 0 uses all available cores
# --halt soon,fail=1 stops on first failure
# --progress shows progress bar
# --joblog creates a log of job execution
JOBLOG_FILE="${WORKDIR}/ParticleNet/logs/multiclass_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$JOBLOG_FILE")"

if command -v parallel > /dev/null; then
    parallel --jobs 0 --halt soon,fail=1 --progress --joblog "$JOBLOG_FILE" < "$TEMP_CMD_FILE"
    exit_code=$?
else
    echo "WARNING: GNU parallel not found, running sequentially"
    exit_code=0
    while IFS= read -r cmd; do
        echo "Executing: $cmd"
        if ! eval "$cmd"; then
            exit_code=1
            break
        fi
    done < "$TEMP_CMD_FILE"
fi

# Clean up
rm -f "$TEMP_CMD_FILE"

# Report results
echo ""
echo "=========================================="
if [[ $exit_code -eq 0 ]]; then
    echo "All training jobs completed successfully!"
    if [[ "$SEPARATE_BJETS" == true ]]; then
        echo "Results saved in: $WORKDIR/ParticleNet/results_bjets/multiclass/$CHANNEL/"
    else
        echo "Results saved in: $WORKDIR/ParticleNet/results/multiclass/$CHANNEL/"
    fi

    # Count total models trained
    total_models=$((${#SIGNAL_ARRAY[@]} * 5))
    echo "Total models trained: $total_models"
else
    echo "Some training jobs failed!"
    echo "Check the job log: $JOBLOG_FILE"
fi

echo "Job log: $JOBLOG_FILE"
echo "=========================================="

exit $exit_code