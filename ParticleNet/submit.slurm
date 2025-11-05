#!/bin/bash
#SBATCH --job-name=GA_optim
#SBATCH --account=m2612
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/slurm-%j.out
#SBATCH --error=logs/slurmslurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=choij@cern.ch

# Parse command-line arguments
CONFIG=${1}
DEVICES=${2:-cuda:0,cuda:1}

if [ -z "$CONFIG" ]; then
    echo "ERROR: Configuration required"
    echo "Usage: sbatch submit_ga.slurm \"sig1:ch1,sig2:ch2\" \"dev1,dev2\""
    echo "Example: sbatch submit_ga.slurm \"MHc130_MA90:Run1E2Mu,MHc130_MA90:Run3Mu\" \"cuda:0,cuda:1\""
    exit 1
fi

# Additional flags (--pilot, --debug)
shift 2
EXTRA_FLAGS="$@"

# Job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID:        $SLURM_JOB_ID"
echo "Job Name:      $SLURM_JOB_NAME"
echo "Node:          $SLURM_NODELIST"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo "=========================================="
echo "Configuration: $CONFIG"
echo "Devices:       $DEVICES"
echo "Extra Flags:   $EXTRA_FLAGS"
echo "=========================================="

# Load modules
echo "Loading modules..."
module load python

# Navigate to ParticleNet directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Setup environment - CRITICAL: source parent setup.sh
echo "Setting up environment..."
cd ..
source setup.sh
cd ParticleNet

# Create logs directory
mkdir -p logs/slurm

# Run GA optimization
echo "Starting GA optimization..."
./scripts/launchGAOptim.sh --config "$CONFIG" --device "$DEVICES" $EXTRA_FLAGS

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "GA optimization completed successfully"
    echo "Completed:     $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: GA optimization failed with exit code $EXIT_CODE"
    echo "Failed:        $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi
