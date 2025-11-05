#!/bin/bash

# Master script for ParticleNet methodology comparison experiment
# Provides easy interface to run the complete experiment

set -e  # Exit on error

# Setup environment
export WORKDIR="${WORKDIR:-${PWD}/..}"
cd "$(dirname "$0")"

echo "=========================================="
echo "ParticleNet Methodology Comparison"
echo "=========================================="
echo "WORKDIR: $WORKDIR"
echo "ParticleNet: $(pwd)"
echo ""

# Function to display usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --channel CHANNEL     Channel for training: Run1E2Mu or Run3Mu (default: Run3Mu)"
    echo "  --cleanup             Clean up existing results before starting"
    echo "  --multiclass-only     Run only multi-class training (3 models)"
    echo "  --binary-only         Run only binary training (9 models)"
    echo "  --pilot               Use pilot datasets for quick testing"
    echo "  --separate_bjets      Use separate b-jets as distinct particles"
    echo "  --debug               Enable debug logging"
    echo "  --dry-run             Show what would be executed"
    echo "  --continue-on-failure Continue even if some training jobs fail"
    echo "  --skip-validation     Skip dataset validation"
    echo "  --max-jobs N          Maximum parallel jobs (0 = use all cores)"
    echo "  --no-parallel         Disable parallel execution (run sequentially)"
    echo "  --job-timeout N       Timeout per job in seconds (default: 3600)"
    echo "  --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Full experiment (12 models)"
    echo "  $0"
    echo ""
    echo "  # Clean up and run full experiment"
    echo "  $0 --cleanup"
    echo ""
    echo "  # Test with pilot data"
    echo "  $0 --pilot --dry-run"
    echo ""
    echo "  # Run only multi-class comparison"
    echo "  $0 --multiclass-only"
    echo ""
    echo "  # Run with 4 parallel jobs"
    echo "  $0 --max-jobs 4"
    echo ""
    echo "  # Run sequentially (no parallelism)"
    echo "  $0 --no-parallel"
    echo ""
    echo "  # Compare standard vs separate b-jets methodology"
    echo "  $0 --separate_bjets --pilot"
    echo ""
    echo "  # Run with Run1E2Mu channel"
    echo "  $0 --channel Run1E2Mu"
    echo ""
    echo "  # Run with Run1E2Mu channel and separate b-jets"
    echo "  $0 --channel Run1E2Mu --separate_bjets"
    echo ""
    echo "Experiment Details:"
    echo "  - Signal points: (MHc, MA) = (160, 85), (130, 90), (100, 95)"
    echo "  - Backgrounds: nonprompt, diboson, ttZ"
    echo "  - Multi-class: 3 models (signal vs. all backgrounds)"
    echo "  - Binary: 9 models (signal vs. each background)"
    echo "  - Training: fold 3 only (for rapid testing)"
    echo "  - Total: 12 training jobs"
}

# Default options
CLEANUP=false
EXPERIMENT_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --channel)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --channel $2"
            shift 2
            ;;
        --cleanup)
            CLEANUP=true
            shift
            ;;
        --multiclass-only)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --multiclass-only"
            shift
            ;;
        --binary-only)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --binary-only"
            shift
            ;;
        --pilot)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --pilot"
            shift
            ;;
        --separate_bjets)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --separate_bjets"
            shift
            ;;
        --debug)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --debug"
            shift
            ;;
        --dry-run)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --dry-run"
            shift
            ;;
        --continue-on-failure)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --continue-on-failure"
            shift
            ;;
        --skip-validation)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --skip-dataset-validation"
            shift
            ;;
        --max-jobs)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --max-parallel-jobs $2"
            shift 2
            ;;
        --no-parallel)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --no-parallel"
            shift
            ;;
        --job-timeout)
            EXPERIMENT_ARGS="$EXPERIMENT_ARGS --job-timeout $2"
            shift 2
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

# Step 1: Cleanup if requested
if [[ "$CLEANUP" == true ]]; then
    echo "Step 1: Cleaning up existing results..."
    if [[ -f "scripts/cleanup_results.sh" ]]; then
        ./scripts/cleanup_results.sh
    else
        echo "WARNING: Cleanup script not found, skipping cleanup"
    fi
    echo ""
fi

# Step 2: Validate environment
echo "Step 2: Environment validation..."
if [[ ! -f "python/run_comparison_experiment.py" ]]; then
    echo "ERROR: Experiment orchestrator not found"
    echo "Expected: python/run_comparison_experiment.py"
    exit 1
fi

# Check if we can import required modules
cd python
if ! python -c "import experiment_config; print('✓ Experiment config loaded')" 2>/dev/null; then
    echo "ERROR: Failed to load experiment configuration"
    echo "Make sure you're in the ParticleNet directory and setup.sh has been sourced"
    exit 1
fi
cd ..

echo "✓ Environment validation passed"
echo ""

# Step 3: Show experiment configuration
echo "Step 3: Experiment configuration..."
python python/experiment_config.py
echo ""

# Step 4: Execute experiment
echo "Step 4: Launching methodology comparison experiment..."
echo "Command: python python/run_comparison_experiment.py $EXPERIMENT_ARGS"
echo ""

# Add path for training scripts
export PATH="${PWD}/python:${PATH}"

# Execute the experiment
python python/run_comparison_experiment.py $EXPERIMENT_ARGS
exit_code=$?

echo ""
echo "=========================================="
if [[ $exit_code -eq 0 ]]; then
    echo "✓ Methodology comparison experiment completed successfully!"
    echo ""
    echo "Results location:"
    if [[ "$EXPERIMENT_ARGS" == *"--separate_bjets"* ]]; then
        echo "  $WORKDIR/ParticleNet/results_bjets/comparison_experiment/"
    else
        echo "  $WORKDIR/ParticleNet/results/comparison_experiment/"
    fi
    echo ""
    echo "Next steps:"
    echo "  1. Analyze results with comparison tools"
    echo "  2. Generate performance plots"
    echo "  3. Compare methodology effectiveness"
else
    echo "✗ Methodology comparison experiment failed!"
    echo ""
    echo "Check logs for details:"
    echo "  $WORKDIR/ParticleNet/logs/"
fi
echo "=========================================="

exit $exit_code
