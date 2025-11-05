#!/bin/bash

# Genetic Algorithm optimization launcher for ParticleNet hyperparameter tuning
# Uses shared memory to efficiently train multiple models in parallel
#
# Usage (Format 1 - paired config):
#   ./scripts/launchGAOptim.sh --config <sig1:ch1,sig2:ch2,...> --device <dev1,dev2,...> [--pilot] [--debug]
#   Example: ./scripts/launchGAOptim.sh --config MHc130_MA100:Run1E2Mu,MHc130_MA90:Run3Mu --device cuda:0,cuda:1
#
# Usage (Format 2 - single signal, multi-channel):
#   ./scripts/launchGAOptim.sh --signal <signal> --channel <ch1,ch2,...> --device <dev1,dev2,...> [--pilot] [--debug]
#   Example: ./scripts/launchGAOptim.sh --signal MHc130_MA100 --channel Run1E2Mu,Run3Mu --device cuda:0,cuda:1
#
# Usage (Format 3 - backward compatible):
#   ./scripts/launchGAOptim.sh <signal> <channel> <device> [--pilot] [--debug]
#   Example: ./scripts/launchGAOptim.sh MHc130_MA100 Run1E2Mu cuda:0 --pilot

print_usage() {
    echo "Error: Missing or invalid arguments"
    echo ""
    echo "GA Optimization Launcher"
    echo ""
    echo "Usage (Format 1 - paired signal:channel config):"
    echo "  $0 --config <sig1:ch1,sig2:ch2,...> --device <dev1,dev2,...> [--pilot] [--debug]"
    echo "  Example: $0 --config MHc130_MA100:Run1E2Mu,MHc130_MA90:Run3Mu --device cuda:0,cuda:1"
    echo ""
    echo "Usage (Format 2 - single signal, multi-channel):"
    echo "  $0 --signal <signal> --channel <ch1,ch2,...> --device <dev1,dev2,...> [--pilot] [--debug]"
    echo "  Example: $0 --signal MHc130_MA100 --channel Run1E2Mu,Run3Mu --device cuda:0,cuda:1"
    echo ""
    echo "Usage (Format 3 - backward compatible):"
    echo "  $0 <signal> <channel> <device> [--pilot] [--debug]"
    echo "  Example: $0 MHc130_MA100 Run1E2Mu cuda:0 --pilot"
}

# Detect if using old positional syntax or new flag syntax
if [[ "$1" != --* ]]; then
    # Format 3: Old positional syntax
    SIGNAL=$1
    CHANNELS=$2
    DEVICES=$3
    shift 3
    EXTRA_FLAGS=("$@")
    USE_CONFIG_MODE=false
else
    # Format 1 or 2: New flag-based syntax
    SIGNAL=""
    CHANNELS=""
    DEVICES=""
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
            --device)
                DEVICES="$2"
                shift 2
                ;;
            --pilot|--debug)
                EXTRA_FLAGS+=("$1")
                shift
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
    if [ -z "$CONFIG" ] || [ -z "$DEVICES" ]; then
        echo "Error: --config and --device are required"
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

    # Split devices
    IFS=',' read -ra DEVICE_ARRAY <<< "$DEVICES"

    # Validate counts match
    if [ ${#CONFIG_ARRAY[@]} -ne ${#DEVICE_ARRAY[@]} ]; then
        echo "Error: Number of configs (${#CONFIG_ARRAY[@]}) must match number of devices (${#DEVICE_ARRAY[@]})"
        echo "Configs: ${CONFIG_ARRAY[*]}"
        echo "Devices: ${DEVICE_ARRAY[*]}"
        exit 1
    fi

else
    # Format 2 or 3: Single signal with channel(s)
    if [ -z "$SIGNAL" ] || [ -z "$CHANNELS" ] || [ -z "$DEVICES" ]; then
        print_usage
        exit 1
    fi

    # Split comma-separated channels and devices into arrays
    IFS=',' read -ra CHANNEL_ARRAY <<< "$CHANNELS"
    IFS=',' read -ra DEVICE_ARRAY <<< "$DEVICES"

    # Create signal array (same signal for all channels)
    SIGNAL_ARRAY=()
    for ch in "${CHANNEL_ARRAY[@]}"; do
        SIGNAL_ARRAY+=("$SIGNAL")
    done

    # Validate that number of channels matches number of devices
    if [ ${#CHANNEL_ARRAY[@]} -ne ${#DEVICE_ARRAY[@]} ]; then
        echo "Error: Number of channels (${#CHANNEL_ARRAY[@]}) must match number of devices (${#DEVICE_ARRAY[@]})"
        echo "Channels: ${CHANNEL_ARRAY[*]}"
        echo "Devices: ${DEVICE_ARRAY[*]}"
        exit 1
    fi
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

# Launch GA optimization for each (signal, channel, device) tuple
PIDS=()
NUM_JOBS=${#SIGNAL_ARRAY[@]}

echo "=========================================="
echo "GA OPTIMIZATION"
echo "=========================================="
echo "Launching ${NUM_JOBS} parallel GA optimization job(s)"
echo "=========================================="

for i in "${!SIGNAL_ARRAY[@]}"; do
    SIGNAL="${SIGNAL_ARRAY[$i]}"
    CHANNEL="${CHANNEL_ARRAY[$i]}"
    DEVICE="${DEVICE_ARRAY[$i]}"
    LOGFILE="logs/GA_${SIGNAL}_${CHANNEL}.log"

    echo "[$((i+1))/${NUM_JOBS}] Signal: ${SIGNAL}, Channel: ${CHANNEL}, Device: ${DEVICE}"
    echo "        Log: ${LOGFILE}"

    # Launch GA optimization
    launchGAOptim.py --signal "$SIGNAL" --channel "$CHANNEL" --device "$DEVICE" "${EXTRA_FLAGS[@]}" \
        > "$LOGFILE" 2>&1 &

    PIDS+=($!)

    # Small delay to avoid resource conflicts at startup
    sleep 2
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
    echo "All ${NUM_JOBS} GA optimization job(s) completed successfully"
    echo "=========================================="
    exit 0
else
    echo "ERROR: ${FAILED}/${NUM_JOBS} job(s) failed"
    echo "Check log files in logs/ directory for details"
    echo "=========================================="
    exit 1
fi
