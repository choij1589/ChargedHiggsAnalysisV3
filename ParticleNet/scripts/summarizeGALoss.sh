#!/bin/bash

# GA Loss Summarization wrapper for batch processing
# Reads GA optimization results and generates loss evolution summaries
#
# Usage (Format 1 - paired config):
#   ./scripts/summarizeGALoss.sh --config <sig1:ch1,sig2:ch2,...> [--input <dir>]
#   Example: ./scripts/summarizeGALoss.sh --config MHc130_MA100:Run1E2Mu,MHc130_MA90:Run3Mu
#
# Usage (Format 2 - single signal, multi-channel):
#   ./scripts/summarizeGALoss.sh --signal <signal> --channel <ch1,ch2,...> [--input <dir>]
#   Example: ./scripts/summarizeGALoss.sh --signal MHc130_MA100 --channel Run1E2Mu,Run3Mu
#
# Usage (Format 3 - backward compatible):
#   ./scripts/summarizeGALoss.sh <signal> <channel> [--input <dir>]
#   Example: ./scripts/summarizeGALoss.sh MHc130_MA100 Run1E2Mu

print_usage() {
    echo "Error: Missing or invalid arguments"
    echo ""
    echo "GA Loss Summarization"
    echo ""
    echo "Usage (Format 1 - paired signal:channel config):"
    echo "  $0 --config <sig1:ch1,sig2:ch2,...> [--input <dir>]"
    echo "  Example: $0 --config MHc130_MA100:Run1E2Mu,MHc130_MA90:Run3Mu"
    echo ""
    echo "Usage (Format 2 - single signal, multi-channel):"
    echo "  $0 --signal <signal> --channel <ch1,ch2,...> [--input <dir>]"
    echo "  Example: $0 --signal MHc130_MA100 --channel Run1E2Mu,Run3Mu"
    echo ""
    echo "Usage (Format 3 - backward compatible):"
    echo "  $0 <signal> <channel> [--input <dir>]"
    echo "  Example: $0 MHc130_MA100 Run1E2Mu --input GAOptim_bjets_maxepoch50"
}

# Detect if using old positional syntax or new flag syntax
if [[ "$1" != --* ]]; then
    # Format 3: Old positional syntax
    SIGNAL=$1
    CHANNELS=$2
    shift 2
    EXTRA_FLAGS=("$@")
    USE_CONFIG_MODE=false
else
    # Format 1 or 2: New flag-based syntax
    SIGNAL=""
    CHANNELS=""
    CONFIG=""
    EXTRA_FLAGS=()
    USE_CONFIG_MODE=false

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
            --channel)
                CHANNELS="$2"
                shift 2
                ;;
            --input)
                EXTRA_FLAGS+=("$1" "$2")
                shift 2
                ;;
            *)
                echo "Error: Unknown option $1"
                print_usage
                exit 1
                ;;
        esac
    done

    # Validate mutually exclusive options
    if [ "$USE_CONFIG_MODE" = true ] && [ -n "$SIGNAL" ]; then
        echo "Error: Cannot use --config together with --signal"
        print_usage
        exit 1
    fi

    if [ "$USE_CONFIG_MODE" = true ] && [ -n "$CHANNELS" ]; then
        echo "Error: Cannot use --config together with --channel"
        print_usage
        exit 1
    fi
fi

# Add python directory to PATH
export PATH=$PATH:$PWD/python

# Parse configurations based on mode
if [ "$USE_CONFIG_MODE" = true ]; then
    # Format 1: Parse signal:channel pairs
    if [ -z "$CONFIG" ]; then
        echo "Error: --config is required"
        print_usage
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
            echo "Error: Invalid config format '$cfg'. Expected format: signal:channel"
            echo "Example: MHc130_MA100:Run1E2Mu"
            exit 1
        fi

        # Split by colon
        IFS=':' read -r sig ch <<< "$cfg"

        # Validate signal is not empty
        if [ -z "$sig" ]; then
            echo "Error: Empty signal in config '$cfg'"
            exit 1
        fi

        # Validate channel
        if [ -z "$ch" ]; then
            echo "Error: Empty channel in config '$cfg'"
            exit 1
        fi

        if [[ ! "$ch" =~ ^(Run1E2Mu|Run3Mu|Combined)$ ]]; then
            echo "Error: Invalid channel '$ch' in config '$cfg'"
            echo "Valid channels: Run1E2Mu, Run3Mu, Combined"
            exit 1
        fi

        SIGNAL_ARRAY+=("$sig")
        CHANNEL_ARRAY+=("$ch")
    done

else
    # Format 2 or 3: Single signal with channel(s)
    if [ -z "$SIGNAL" ] || [ -z "$CHANNELS" ]; then
        print_usage
        exit 1
    fi

    # Split comma-separated channels into arrays
    IFS=',' read -ra CHANNEL_ARRAY <<< "$CHANNELS"

    # Create signal array (same signal for all channels)
    SIGNAL_ARRAY=()
    for ch in "${CHANNEL_ARRAY[@]}"; do
        SIGNAL_ARRAY+=("$SIGNAL")
    done
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Setup trap to kill all background jobs on script interruption (Ctrl+C, SIGTERM, etc.)
cleanup() {
    echo ""
    echo "=========================================="
    echo "Script interrupted! Cleaning up..."
    echo "=========================================="
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo "Killing ${#PIDS[@]} background job(s)..."
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Killing PID $pid"
                kill "$pid" 2>/dev/null
            fi
        done
        # Wait a moment for graceful shutdown
        sleep 1
        # Force kill any remaining processes
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing PID $pid"
                kill -9 "$pid" 2>/dev/null
            fi
        done
    fi
    echo "Cleanup complete"
    exit 130
}

trap cleanup SIGINT SIGTERM

# Launch summarization for each (signal, channel) tuple
PIDS=()
NUM_JOBS=${#SIGNAL_ARRAY[@]}

echo "=========================================="
echo "GA LOSS SUMMARIZATION"
echo "=========================================="
echo "Processing ${NUM_JOBS} signal/channel combination(s)"
echo "=========================================="

for i in "${!SIGNAL_ARRAY[@]}"; do
    SIGNAL="${SIGNAL_ARRAY[$i]}"
    CHANNEL="${CHANNEL_ARRAY[$i]}"
    LOGFILE="logs/GALossSummary_${SIGNAL}_${CHANNEL}.log"

    echo "[$((i+1))/${NUM_JOBS}] Signal: ${SIGNAL}, Channel: ${CHANNEL}"
    echo "        Log: ${LOGFILE}"

    # Launch summarization
    summarizeGALoss.py --signal "$SIGNAL" --channel "$CHANNEL" "${EXTRA_FLAGS[@]}" \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)

    # Small delay to avoid resource conflicts at startup
    sleep 1
done

echo "=========================================="
echo "All jobs launched. Waiting for completion..."
echo "=========================================="

# Wait for all background processes and track failures
FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    SIGNAL="${SIGNAL_ARRAY[$i]}"
    CHANNEL="${CHANNEL_ARRAY[$i]}"

    wait $PID
    EXIT_CODE=$?

    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: ${SIGNAL}/${CHANNEL} (PID ${PID}) failed with exit code ${EXIT_CODE}"
        FAILED=$((FAILED + 1))
    else
        echo "SUCCESS: ${SIGNAL}/${CHANNEL} (PID ${PID}) completed"
    fi
done

echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "All ${NUM_JOBS} summarization job(s) completed successfully"
    echo "=========================================="
    exit 0
else
    echo "ERROR: ${FAILED}/${NUM_JOBS} job(s) failed"
    echo "Check log files in logs/ directory for details"
    echo "=========================================="
    exit 1
fi
