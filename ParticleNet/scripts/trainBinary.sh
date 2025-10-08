#!/bin/bash

# Binary training script for ParticleNet methodology comparison
# Trains signal vs. single background models for fold 3 only

set -e  # Exit on error

# Setup environment
export PATH="${PWD}/python:${PATH}"
export WORKDIR="${WORKDIR:-${PWD}/../..}"

# Check if we're in the right directory
if [[ ! -f "python/trainBinary.py" ]]; then
    echo "ERROR: Must run from ParticleNet directory"
    echo "Expected to find python/trainBinary.py"
    exit 1
fi

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --channel CHANNEL     Channel to train (Run1E2Mu, Run3Mu) [default: Run1E2Mu]"
    echo "  --signal SIGNAL       Signal sample name without prefix [required]"
    echo "  --background BG       Background category (nonprompt, diboson, ttZ) [required]"
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
    echo "  --all-scenarios       Train all binary scenarios from experiment config"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Single binary training"
    echo "  $0 --signal MHc160_MA85 --background nonprompt"
    echo "  # All binary scenarios"
    echo "  $0 --all-scenarios"
    echo "  # Different model configuration"
    echo "  $0 --signal MHc130_MA90 --background diboson --model ParticleNetV2"
}

# Default parameters
CHANNEL="Run1E2Mu"
SIGNAL=""
BACKGROUND=""
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
ALL_SCENARIOS=false
FOLD=3  # Fixed to fold 3 for methodology comparison

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --channel)
            CHANNEL="$2"
            shift 2
            ;;
        --signal)
            SIGNAL="$2"
            shift 2
            ;;
        --background)
            BACKGROUND="$2"
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
        --all-scenarios)
            ALL_SCENARIOS=true
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

# Validate channel
if [[ "$CHANNEL" != "Run1E2Mu" && "$CHANNEL" != "Run3Mu" && "$CHANNEL" != "Combined" ]]; then
    echo "ERROR: Invalid channel '$CHANNEL'. Must be Run1E2Mu, Run3Mu, or Combined"
    exit 1
fi

# Build base training command
BASE_CMD="python trainBinary.py"
BASE_CMD="$BASE_CMD --channel $CHANNEL"
BASE_CMD="$BASE_CMD --fold $FOLD"
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

# Handle single scenario vs all scenarios
if [[ "$ALL_SCENARIOS" == true ]]; then
    echo "=========================================="
    echo "Binary Training - All Scenarios (Fold $FOLD)"
    echo "=========================================="
    echo "Channel: $CHANNEL"
    echo "Model: $MODEL ($NNODES nodes)"
    echo "Optimization: $OPTIMIZER (LR: $INIT_LR, decay: 1e-4)"
    echo "Schedule: $SCHEDULER, Loss: $LOSS_TYPE"
    echo "Max epochs: $MAX_EPOCHS"
    echo "Pilot mode: $PILOT"
    echo "Separate b-jets: $SEPARATE_BJETS"
    echo "Debug mode: $DEBUG"
    echo "=========================================="

    # Define all scenarios from experiment config
    SIGNALS=("MHc160_MA85" "MHc130_MA90" "MHc100_MA95")
    BACKGROUNDS=("nonprompt" "diboson" "ttZ")

    # Generate all training commands
    COMMANDS=()
    for signal in "${SIGNALS[@]}"; do
        for background in "${BACKGROUNDS[@]}"; do
            cmd="$BASE_CMD --signal $signal --background $background"
            COMMANDS+=("$cmd")
        done
    done

    echo "Generated ${#COMMANDS[@]} binary training commands"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN - Commands that would be executed:"
        echo "=========================================="
        for i in "${!COMMANDS[@]}"; do
            printf "%2d: %s\n" $((i+1)) "${COMMANDS[$i]}"
        done
        echo ""
        echo "To execute, run without --dry-run"
        exit 0
    fi

    # Execute commands in parallel or sequentially
    echo "Starting binary training for all scenarios..."

    # Create a temporary file with all commands
    TEMP_CMD_FILE=$(mktemp)
    printf "%s\n" "${COMMANDS[@]}" > "$TEMP_CMD_FILE"

    # Create log directory
    JOBLOG_FILE="${WORKDIR}/ParticleNet/logs/binary_training_$(date +%Y%m%d_%H%M%S).log"
    mkdir -p "$(dirname "$JOBLOG_FILE")"

    if command -v parallel > /dev/null; then
        echo "Using GNU parallel for execution..."
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
        echo "All binary training jobs completed successfully!"
        if [[ "$SEPARATE_BJETS" == true ]]; then
            echo "Results saved in: $WORKDIR/ParticleNet/results_bjets/binary/$CHANNEL/"
        else
            echo "Results saved in: $WORKDIR/ParticleNet/results/binary/$CHANNEL/"
        fi
        echo "Total models trained: ${#COMMANDS[@]}"
    else
        echo "Some binary training jobs failed!"
        echo "Check the job log: $JOBLOG_FILE"
    fi
    echo "Job log: $JOBLOG_FILE"
    echo "=========================================="

    exit $exit_code

else
    # Single scenario training
    if [[ -z "$SIGNAL" ]]; then
        echo "ERROR: --signal is required for single scenario training"
        usage
        exit 1
    fi

    if [[ -z "$BACKGROUND" ]]; then
        echo "ERROR: --background is required for single scenario training"
        usage
        exit 1
    fi

    # Validate background category
    if [[ "$BACKGROUND" != "nonprompt" && "$BACKGROUND" != "diboson" && "$BACKGROUND" != "ttZ" ]]; then
        echo "ERROR: Invalid background '$BACKGROUND'. Must be nonprompt, diboson, or ttZ"
        exit 1
    fi

    echo "=========================================="
    echo "Binary Training - Single Scenario"
    echo "=========================================="
    echo "Signal: $SIGNAL"
    echo "Background: $BACKGROUND"
    echo "Channel: $CHANNEL, Fold: $FOLD"
    echo "Model: $MODEL ($NNODES nodes)"
    echo "Optimization: $OPTIMIZER (LR: $INIT_LR)"
    echo "Schedule: $SCHEDULER, Loss: $LOSS_TYPE"
    echo "Max epochs: $MAX_EPOCHS"
    echo "Pilot mode: $PILOT"
    echo "Separate b-jets: $SEPARATE_BJETS"
    echo "Debug mode: $DEBUG"
    echo "=========================================="

    # Build final command
    FINAL_CMD="$BASE_CMD --signal $SIGNAL --background $BACKGROUND"

    if [[ "$DRY_RUN" == true ]]; then
        echo "DRY RUN - Command that would be executed:"
        echo "$FINAL_CMD"
        exit 0
    fi

    echo "Starting binary training..."
    echo "Command: $FINAL_CMD"
    echo ""

    # Execute training
    eval "$FINAL_CMD"
    exit_code=$?

    echo ""
    echo "=========================================="
    if [[ $exit_code -eq 0 ]]; then
        echo "Binary training completed successfully!"
        echo "Signal vs. Background: $SIGNAL vs. $BACKGROUND"
    else
        echo "Binary training failed!"
    fi
    echo "=========================================="

    exit $exit_code
fi