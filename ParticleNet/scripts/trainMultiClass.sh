#!/bin/bash

# Multi-class training script for ParticleNet
# Trains 4-class models (signal vs 3 backgrounds) across all folds in parallel
#
# Usage (Format 1 - paired signal:channel config):
#   ./scripts/trainMultiClass.sh --config <sig1:ch1,sig2:ch2,...> [--param FILE] [--dry-run]
#   Example: ./scripts/trainMultiClass.sh --config MHc130_MA100:Run1E2Mu,MHc130_MA90:Run3Mu
#
# Usage (Format 2 - single signal, multi-channel):
#   ./scripts/trainMultiClass.sh --signal <signal> --channels <ch1,ch2,...> [--param FILE] [--dry-run]
#   Example: ./scripts/trainMultiClass.sh --signal MHc130_MA100 --channels Run1E2Mu,Run3Mu
#
# Usage (Format 3 - single channel, multi-signal):
#   ./scripts/trainMultiClass.sh --channel <channel> --signals <sig1,sig2,...> [--param FILE] [--dry-run]
#   Example: ./scripts/trainMultiClass.sh --channel Run1E2Mu --signals MHc130_MA100,MHc130_MA90

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
    echo "Multi-Class ParticleNet Training"
    echo ""
    echo "Usage (Format 1 - paired signal:channel config):"
    echo "  $0 --config <sig1:ch1,sig2:ch2,...> [--param FILE] [--dry-run]"
    echo "  Example: $0 --config MHc130_MA100:Run1E2Mu,MHc130_MA90:Run3Mu"
    echo ""
    echo "Usage (Format 2 - single signal, multi-channel):"
    echo "  $0 --signal <signal> --channels <ch1,ch2,...> [--param FILE] [--dry-run]"
    echo "  Example: $0 --signal MHc130_MA100 --channels Run1E2Mu,Run3Mu"
    echo ""
    echo "Usage (Format 3 - single channel, multi-signal):"
    echo "  $0 --channel <channel> --signals <sig1,sig2,...> [--param FILE] [--dry-run]"
    echo "  Example: $0 --channel Run1E2Mu --signals MHc130_MA100,MHc130_MA90"
    echo ""
    echo "Options:"
    echo "  --config CONFIG       Comma-separated signal:channel pairs (Format 1)"
    echo "  --signal SIGNAL       Single signal for multi-channel (Format 2)"
    echo "  --signals SIGNALS     Comma-separated signals for single channel (Format 3)"
    echo "  --channel CHANNEL     Single channel (Format 3)"
    echo "  --channels CHANNELS   Comma-separated channels (Format 2)"
    echo "  --param FILE          Path to parameter JSON file [default: configs/SglConfig.json]"
    echo "  --dry-run             Show commands that would be executed"
    echo "  --help                Show this help message"
    echo ""
    echo "Configuration:"
    echo "  All training parameters (model, optimizer, learning rate, backgrounds, etc.)"
    echo "  are configured in configs/SglConfig.json"
    echo ""
    echo "  To customize training, create a custom parameter file:"
    echo "    cp configs/SglConfig.json configs/SglConfig-custom.json"
    echo "    # Edit configs/SglConfig-custom.json"
    echo "    $0 --config MHc130_MA100:Run1E2Mu --param configs/SglConfig-custom.json"
}

# Default parameters
SIGNAL=""
SIGNALS=""
CHANNEL=""
CHANNELS=""
CONFIG=""
PARAM_FILE=""
DRY_RUN=false
USE_CONFIG_MODE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            USE_CONFIG_MODE=true
            shift 2
            ;;
        --signal)
            SIGNAL="$2"
            shift 2
            ;;
        --signals)
            SIGNALS="$2"
            shift 2
            ;;
        --channel)
            CHANNEL="$2"
            shift 2
            ;;
        --channels)
            CHANNELS="$2"
            shift 2
            ;;
        --param)
            PARAM_FILE="$2"
            shift 2
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

# Validate mutually exclusive options
if [ "$USE_CONFIG_MODE" = true ]; then
    # Format 1: --config mode
    if [ -n "$SIGNAL" ] || [ -n "$SIGNALS" ]; then
        echo "ERROR: Cannot use --config together with --signal or --signals"
        usage
        exit 1
    fi
    if [ -n "$CHANNEL" ] || [ -n "$CHANNELS" ]; then
        echo "ERROR: Cannot use --config together with --channel or --channels"
        usage
        exit 1
    fi
else
    # Format 2 or 3: validate mutual exclusivity
    if [ -n "$SIGNAL" ] && [ -n "$SIGNALS" ]; then
        echo "ERROR: Cannot use --signal and --signals together"
        usage
        exit 1
    fi
    if [ -n "$CHANNEL" ] && [ -n "$CHANNELS" ]; then
        echo "ERROR: Cannot use --channel and --channels together"
        usage
        exit 1
    fi

    # Format 2: --signal with --channels
    if [ -n "$SIGNAL" ] && [ -n "$CHANNELS" ]; then
        if [ -n "$CHANNEL" ] || [ -n "$SIGNALS" ]; then
            echo "ERROR: Format 2 uses --signal with --channels only"
            usage
            exit 1
        fi
    # Format 3: --channel with --signals
    elif [ -n "$CHANNEL" ]; then
        if [ -n "$SIGNAL" ] || [ -n "$CHANNELS" ]; then
            echo "ERROR: Format 3 uses --channel with --signals only"
            usage
            exit 1
        fi
    else
        echo "ERROR: Must specify either --config (Format 1), --signal with --channels (Format 2), or --channel (Format 3)"
        usage
        exit 1
    fi
fi

# Parse configurations based on mode
if [ "$USE_CONFIG_MODE" = true ]; then
    # Format 1: Parse signal:channel pairs
    if [ -z "$CONFIG" ]; then
        echo "ERROR: --config is required in Format 1"
        usage
        exit 1
    fi

    # Split config by comma
    IFS=',' read -ra CONFIG_ARRAY <<< "$CONFIG"

    # Parse each signal:channel pair
    SIGNAL_ARRAY=()
    CHANNEL_ARRAY=()

    for cfg in "${CONFIG_ARRAY[@]}"; do
        # Validate format: must contain exactly one colon
        if [[ ! "$cfg" =~ ^[^:]+:[^:]+$ ]]; then
            echo "ERROR: Invalid config format '$cfg'. Expected format: signal:channel"
            echo "Example: MHc130_MA100:Run1E2Mu"
            exit 1
        fi

        # Split by colon
        IFS=':' read -r sig ch <<< "$cfg"

        # Validate signal is not empty
        if [ -z "$sig" ]; then
            echo "ERROR: Empty signal in config '$cfg'"
            exit 1
        fi

        # Validate channel
        if [ -z "$ch" ]; then
            echo "ERROR: Empty channel in config '$cfg'"
            exit 1
        fi

        if [[ ! "$ch" =~ ^(Run1E2Mu|Run3Mu|Combined)$ ]]; then
            echo "ERROR: Invalid channel '$ch' in config '$cfg'"
            echo "Valid channels: Run1E2Mu, Run3Mu, Combined"
            exit 1
        fi

        SIGNAL_ARRAY+=("$sig")
        CHANNEL_ARRAY+=("$ch")
    done

elif [ -n "$SIGNAL" ] && [ -n "$CHANNELS" ]; then
    # Format 2: Single signal with multiple channels
    IFS=',' read -ra CHANNEL_ARRAY <<< "$CHANNELS"

    # Validate channels
    for ch in "${CHANNEL_ARRAY[@]}"; do
        if [[ ! "$ch" =~ ^(Run1E2Mu|Run3Mu|Combined)$ ]]; then
            echo "ERROR: Invalid channel '$ch'"
            echo "Valid channels: Run1E2Mu, Run3Mu, Combined"
            exit 1
        fi
    done

    # Create signal array (same signal for all channels)
    SIGNAL_ARRAY=()
    for ch in "${CHANNEL_ARRAY[@]}"; do
        SIGNAL_ARRAY+=("$SIGNAL")
    done

else
    # Format 3: Single channel with multiple signals (backward compatible)
    if [ -z "$CHANNEL" ]; then
        echo "ERROR: --channel is required in Format 3"
        usage
        exit 1
    fi

    # Validate channel
    if [[ ! "$CHANNEL" =~ ^(Run1E2Mu|Run3Mu|Combined)$ ]]; then
        echo "ERROR: Invalid channel '$CHANNEL'. Must be Run1E2Mu, Run3Mu, or Combined"
        exit 1
    fi

    # Default signal samples if not specified
    if [[ -z "$SIGNALS" ]]; then
        SIGNALS="TTToHcToWAToMuMu-MHc100MA95,TTToHcToWAToMuMu-MHc130MA90,TTToHcToWAToMuMu-MHc130MA100,TTToHcToWAToMuMu-MHc160MA85,TTToHcToWAToMuMu-MHc160MA155"
    fi

    # Convert comma-separated signals to array
    IFS=',' read -ra SIGNAL_ARRAY <<< "$SIGNALS"

    # Create channel array (same channel for all signals)
    CHANNEL_ARRAY=()
    for sig in "${SIGNAL_ARRAY[@]}"; do
        CHANNEL_ARRAY+=("$CHANNEL")
    done
fi

# Validate that we have matching arrays
if [ ${#SIGNAL_ARRAY[@]} -ne ${#CHANNEL_ARRAY[@]} ]; then
    echo "ERROR: Signal and channel arrays have mismatched lengths"
    echo "Signals (${#SIGNAL_ARRAY[@]}): ${SIGNAL_ARRAY[*]}"
    echo "Channels (${#CHANNEL_ARRAY[@]}): ${CHANNEL_ARRAY[*]}"
    exit 1
fi

echo "=========================================="
echo "Multi-Class ParticleNet Training"
echo "=========================================="
echo "Number of training jobs: ${#SIGNAL_ARRAY[@]}"
for i in "${!SIGNAL_ARRAY[@]}"; do
    echo "  [$((i+1))] Signal: ${SIGNAL_ARRAY[$i]}, Channel: ${CHANNEL_ARRAY[$i]}"
done
if [[ -n "$PARAM_FILE" ]]; then
    echo "Parameter file: $PARAM_FILE"
else
    echo "Parameter file: configs/SglConfig.json (default)"
fi
echo "Dry run: $DRY_RUN"
echo ""
echo "Note: All training parameters (model, optimizer, backgrounds, etc.)"
echo "      are configured in the JSON parameter file."
echo "=========================================="

# Generate all training commands
COMMANDS=()
for i in "${!SIGNAL_ARRAY[@]}"; do
    sig="${SIGNAL_ARRAY[$i]}"
    ch="${CHANNEL_ARRAY[$i]}"

    cmd="python trainMultiClass.py --signal $sig --channel $ch"

    if [[ -n "$PARAM_FILE" ]]; then
        cmd="$cmd --config $PARAM_FILE"
    fi

    COMMANDS+=("$cmd")
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

# Check if dataset directories exist
echo "Checking dataset availability..."
DATASET_ROOT="$WORKDIR/ParticleNet/dataset/samples"
DATASET_ROOT_BJETS="$WORKDIR/ParticleNet/dataset_bjets/samples"

if [[ ! -d "$DATASET_ROOT" ]] && [[ ! -d "$DATASET_ROOT_BJETS" ]]; then
    echo "ERROR: No dataset directories found"
    echo "  Checked: $DATASET_ROOT"
    echo "  Checked: $DATASET_ROOT_BJETS"
    echo "Please run dataset creation first"
    exit 1
fi

# Check signal-channel combinations (check both regular and bjets datasets)
missing_datasets=()
for i in "${!SIGNAL_ARRAY[@]}"; do
    sig="${SIGNAL_ARRAY[$i]}"
    ch="${CHANNEL_ARRAY[$i]}"

    found=false
    for dataset_root in "$DATASET_ROOT" "$DATASET_ROOT_BJETS"; do
        if [[ -d "$dataset_root/signals/$sig/$ch" ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" == false ]]; then
        missing_datasets+=("$sig:$ch")
    fi
done

if [[ ${#missing_datasets[@]} -gt 0 ]]; then
    echo "ERROR: Missing signal-channel dataset combinations:"
    printf "  %s\n" "${missing_datasets[@]}"
    echo "Please run dataset creation for these signal-channel combinations"
    exit 1
fi

echo "All required signal-channel datasets found"
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

    # Get unique channels
    unique_channels=($(printf "%s\n" "${CHANNEL_ARRAY[@]}" | sort -u))

    echo "Results saved in: $WORKDIR/ParticleNet/results*/multiclass/{CHANNEL}/"
    echo "Channels trained: ${unique_channels[*]}"
    echo "(Check both results/ and results_bjets/ depending on config)"

    # Count total models trained
    total_models=${#SIGNAL_ARRAY[@]}
    echo "Total models trained: $total_models"
    echo ""
    echo "Note: Fold configuration is determined by train_folds/valid_folds/test_folds"
    echo "      in the parameter file. To perform 5-fold CV, run with different configs."
else
    echo "Some training jobs failed!"
    echo "Check the job log: $JOBLOG_FILE"
fi

echo "Job log: $JOBLOG_FILE"
echo "=========================================="

exit $exit_code