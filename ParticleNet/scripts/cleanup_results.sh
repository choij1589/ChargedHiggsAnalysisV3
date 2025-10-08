#!/bin/bash

# Cleanup script for ParticleNet results, logs, and models
# Prepares clean environment for methodology comparison experiment

set -e  # Exit on error

# Setup environment
export WORKDIR="${WORKDIR:-${PWD}/../..}"

echo "=========================================="
echo "ParticleNet Results Cleanup"
echo "=========================================="
echo "WORKDIR: $WORKDIR"
echo ""

# Function to safely remove directory with confirmation
safe_remove() {
    local dir="$1"
    local description="$2"

    if [[ -d "$dir" ]]; then
        echo "Removing $description: $dir"
        rm -rf "$dir"
        echo "  ✓ Removed"
    else
        echo "  ✓ $description not found (already clean)"
    fi
}

# Function to safely remove files matching pattern
safe_remove_files() {
    local pattern="$1"
    local description="$2"

    if ls $pattern 1> /dev/null 2>&1; then
        echo "Removing $description: $pattern"
        rm -f $pattern
        echo "  ✓ Removed"
    else
        echo "  ✓ $description not found (already clean)"
    fi
}

echo "Cleaning ParticleNet directories..."

# Remove results directories (new channel-first structure)
safe_remove "$WORKDIR/ParticleNet/results" "All results (standard)"
safe_remove "$WORKDIR/ParticleNet/results_bjets" "All results (separate b-jets)"

# Remove plots directories
safe_remove "$WORKDIR/ParticleNet/plots" "All plots (standard)"
safe_remove "$WORKDIR/ParticleNet/plots_bjets" "All plots (separate b-jets)"

# Remove logs directory
safe_remove "$WORKDIR/ParticleNet/logs" "Training logs"

# Remove any temporary model files in current directory
cd "$WORKDIR/ParticleNet"
safe_remove_files "*.pt" "PyTorch model files"
safe_remove_files "*.pth" "PyTorch checkpoint files"
safe_remove_files "*.log" "Log files"

# Remove Python cache directories
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
echo "  ✓ Python cache cleaned"

# Remove any Jupyter notebook checkpoints
safe_remove ".ipynb_checkpoints" "Jupyter checkpoints"

echo ""
echo "=========================================="
echo "Cleanup completed successfully!"
echo "=========================================="
echo ""
echo "Cleaned directories:"
echo "  - ParticleNet/results/ (all channels, binary & multiclass)"
echo "  - ParticleNet/results_bjets/ (separate b-jets results)"
echo "  - ParticleNet/plots/ (all visualization outputs)"
echo "  - ParticleNet/plots_bjets/ (separate b-jets plots)"
echo "  - ParticleNet/logs/"
echo "  - Python cache files (__pycache__, *.pyc)"
echo "  - Temporary model files (*.pt, *.pth, *.log)"
echo ""
echo "Ready for fresh methodology comparison experiment!"