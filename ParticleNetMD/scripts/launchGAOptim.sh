#!/bin/bash

# Genetic Algorithm optimization launcher for ParticleNetMD hyperparameter tuning
# Uses shared memory to efficiently train multiple models in parallel with DisCo loss
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
#
# Profiling (any format): add --perf to wrap each job in `perf record -F 99 -g` with
# PYTHONPERFSUPPORT=1 (Python 3.12+), so Python function names appear in the stacks.
# Spawned workers inherit the env var, and perf follows child processes by default.
#   Outputs: profiling/perf/GA_<signal>_<channel>.{perf.data,folded,svg}
#   FlameGraph scripts: $FLAMEGRAPH_DIR (default: ./FlameGraph,
#   git clone https://github.com/brendangregg/FlameGraph)
#   The flamegraph is built at the end of the run on the same node, because perf script
#   needs the /tmp/perf-<pid>.map files that Python writes there.
#
# Or add --py-spy (mutually exclusive with --perf) to wrap each job in
# `py-spy record --subprocesses --native`. py-spy reads the interpreter state directly,
# so Python frames are resolved without frame pointers, and --native adds the C/C++
# frames (torch, CUDA) below them. Raw folded stacks are rendered with flamegraph.pl.
#   Outputs: profiling/py-spy/GA_<signal>_<channel>.{folded,svg}
#   Sampling rate: $PYSPY_RATE samples/s per process (default 100)

print_usage() {
    echo "Error: Missing or invalid arguments"
    echo ""
    echo "GA Optimization Launcher for ParticleNetMD"
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
    echo ""
    echo "Profiling: add --perf to any format (perf record + PYTHONPERFSUPPORT=1, flamegraph in profiling/perf/)"
    echo "       or: add --py-spy to any format (py-spy record --subprocesses --native, flamegraph in profiling/py-spy/)"
}

PROFILER=""   # "", "perf" or "py-spy"

set_profiler() {
    if [ -n "$PROFILER" ] && [ "$PROFILER" != "$1" ]; then
        echo "Error: --perf and --py-spy are mutually exclusive"
        exit 1
    fi
    PROFILER="$1"
}

# Detect if using old positional syntax or new flag syntax
if [[ "$1" != --* ]]; then
    # Format 3: Old positional syntax
    SIGNAL=$1
    CHANNELS=$2
    DEVICES=$3
    shift 3
    EXTRA_FLAGS=()
    for flag in "$@"; do
        case "$flag" in
            --perf)   set_profiler perf ;;
            --py-spy) set_profiler py-spy ;;
            *)        EXTRA_FLAGS+=("$flag") ;;
        esac
    done
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
            --resume-from)
                EXTRA_FLAGS+=("$1" "$2")
                shift 2
                ;;
            --pilot|--debug)
                EXTRA_FLAGS+=("$1")
                shift
                ;;
            --perf)
                set_profiler perf
                shift
                ;;
            --py-spy)
                set_profiler py-spy
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

# Profiling setup: fail fast before launching hours of training
FLAMEGRAPH_DIR="${FLAMEGRAPH_DIR:-${PWD}/FlameGraph}"
PYSPY_RATE="${PYSPY_RATE:-100}"
if [ -n "$PROFILER" ]; then
    if ! command -v "$PROFILER" > /dev/null 2>&1; then
        echo "Error: --${PROFILER} requested but '${PROFILER}' is not in PATH"
        exit 1
    fi
    if [ ! -x "${FLAMEGRAPH_DIR}/flamegraph.pl" ]; then
        echo "Error: FlameGraph scripts not found in ${FLAMEGRAPH_DIR}"
        echo "  git clone https://github.com/brendangregg/FlameGraph  (or set FLAMEGRAPH_DIR)"
        exit 1
    fi
    mkdir -p "profiling/${PROFILER}"
fi
if [ "$PROFILER" = perf ]; then
    if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)'; then
        echo "Error: PYTHONPERFSUPPORT needs Python >= 3.12, found $(python -V 2>&1)"
        exit 1
    fi
    if [ ! -x "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl" ]; then
        echo "Error: ${FLAMEGRAPH_DIR}/stackcollapse-perf.pl not found"
        exit 1
    fi
    export PYTHONPERFSUPPORT=1
fi
if [ "$PROFILER" = py-spy ] && ! [[ "$PYSPY_RATE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: PYSPY_RATE must be a positive integer, got '${PYSPY_RATE}'"
    exit 1
fi

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
echo "GA OPTIMIZATION (ParticleNetMD)"
echo "=========================================="
echo "Launching ${NUM_JOBS} parallel GA optimization job(s)"
echo "Features: DisCo loss, 16 KS tests, group-balanced accuracy"
echo "=========================================="

for i in "${!SIGNAL_ARRAY[@]}"; do
    SIGNAL="${SIGNAL_ARRAY[$i]}"
    CHANNEL="${CHANNEL_ARRAY[$i]}"
    DEVICE="${DEVICE_ARRAY[$i]}"
    LOGFILE="logs/GA_${SIGNAL}_${CHANNEL}.log"

    echo "[$((i+1))/${NUM_JOBS}] Signal: ${SIGNAL}, Channel: ${CHANNEL}, Device: ${DEVICE}"
    echo "        Log: ${LOGFILE}"

    # Launch GA optimization (optionally under a profiler)
    GA_CMD=(python "$(command -v launchGAOptim.py)" --signal "$SIGNAL" --channel "$CHANNEL" --device "$DEVICE" "${EXTRA_FLAGS[@]}")
    STEM="profiling/${PROFILER}/GA_${SIGNAL}_${CHANNEL}"
    case "$PROFILER" in
        perf)
            echo "        perf: ${STEM}.perf.data"
            perf record -F 99 -g -o "${STEM}.perf.data" -- "${GA_CMD[@]}" > "$LOGFILE" 2>&1 &
            ;;
        py-spy)
            echo "        py-spy: ${STEM}.folded (${PYSPY_RATE} samples/s)"
            py-spy record --subprocesses --native --rate "$PYSPY_RATE" --format raw \
                -o "${STEM}.folded" -- "${GA_CMD[@]}" > "$LOGFILE" 2>&1 &
            ;;
        *)
            "${GA_CMD[@]}" > "$LOGFILE" 2>&1 &
            ;;
    esac

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

# Build flamegraphs now, on this node, while /tmp/perf-<pid>.map still exists.
# Also done for failed jobs: a partial profile is often what explains the failure.
if [ "$PROFILER" = perf ]; then
    echo "=========================================="
    echo "Building flamegraphs (profiling/perf/)"
    echo "=========================================="
    for i in "${!SIGNAL_ARRAY[@]}"; do
        STEM="profiling/perf/GA_${SIGNAL_ARRAY[$i]}_${CHANNEL_ARRAY[$i]}"
        if [ ! -s "${STEM}.perf.data" ]; then
            echo "ERROR: ${STEM}.perf.data missing or empty"
            FAILED=$((FAILED + 1))
            continue
        fi
        if perf script -i "${STEM}.perf.data" 2> "${STEM}.perf_script.log" \
                | "${FLAMEGRAPH_DIR}/stackcollapse-perf.pl" > "${STEM}.folded" \
            && "${FLAMEGRAPH_DIR}/flamegraph.pl" "${STEM}.folded" > "${STEM}.svg"; then
            echo "  ${STEM}.svg"
        else
            echo "ERROR: flamegraph build failed for ${STEM} (see ${STEM}.perf_script.log)"
            FAILED=$((FAILED + 1))
        fi
    done
fi

# py-spy already wrote folded stacks; render them the same way as the perf ones
if [ "$PROFILER" = py-spy ]; then
    echo "=========================================="
    echo "Building flamegraphs (profiling/py-spy/)"
    echo "=========================================="
    for i in "${!SIGNAL_ARRAY[@]}"; do
        STEM="profiling/py-spy/GA_${SIGNAL_ARRAY[$i]}_${CHANNEL_ARRAY[$i]}"
        if [ ! -s "${STEM}.folded" ]; then
            echo "ERROR: ${STEM}.folded missing or empty (see the job log)"
            FAILED=$((FAILED + 1))
            continue
        fi
        if "${FLAMEGRAPH_DIR}/flamegraph.pl" --title "py-spy: GA_${SIGNAL_ARRAY[$i]}_${CHANNEL_ARRAY[$i]}" \
                "${STEM}.folded" > "${STEM}.svg"; then
            echo "  ${STEM}.svg"
        else
            echo "ERROR: flamegraph build failed for ${STEM}"
            FAILED=$((FAILED + 1))
        fi
    done
fi

echo "=========================================="
if [ $FAILED -ne 0 ]; then
    echo "ERROR: ${FAILED}/${NUM_JOBS} job(s) failed"
    echo "Check log files in logs/ directory for details"
    echo "=========================================="
    exit 1
fi

echo "All ${NUM_JOBS} GA optimization job(s) completed successfully"
echo "=========================================="
echo "Next steps: run post-processing (eval / viz / summary) interactively"
echo "  evaluateGAModels.py --signal <signal> --channel <channel> --device <device> --iteration <N>"
echo "  visualizeGAIteration.py --signal <signal> --channel <channel> --device <device> --iteration <N>"
echo "  summarizeGALoss.py --signal <signal> --channel <channel>"
echo "=========================================="
