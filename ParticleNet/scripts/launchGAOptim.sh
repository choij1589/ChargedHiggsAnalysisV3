#!/bin/bash

# Genetic Algorithm optimization launcher for ParticleNet multi-class training
# Usage: ./scripts/launchGAOptim.sh <signal> <channel> <device> [--pilot] [--debug]
# Example: ./scripts/launchGAOptim.sh MHc130_MA100 Run1E2Mu cuda:0
# Example with pilot: ./scripts/launchGAOptim.sh MHc130_MA100 Run1E2Mu cuda:0 --pilot

SIGNAL=$1
CHANNEL=$2
DEVICE=$3
shift 3  # Remove first 3 arguments, remaining are optional flags

# Validate arguments
if [ -z "$SIGNAL" ] || [ -z "$CHANNEL" ] || [ -z "$DEVICE" ]; then
    echo "Usage: $0 <signal> <channel> <device> [--pilot] [--debug]"
    echo "Example: $0 MHc130_MA100 Run1E2Mu cuda:0"
    echo "Example with pilot: $0 MHc130_MA100 Run1E2Mu cuda:0 --pilot"
    exit 1
fi

# Add python directory to PATH
export PATH=$PATH:$PWD/python

# Launch GA optimization (pass through any additional flags)
launchGAOptim.py --signal $SIGNAL --channel $CHANNEL --device $DEVICE "$@"
