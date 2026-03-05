#!/bin/bash
#SBATCH --job-name=lambda_sweep
#SBATCH --account=m2612
#SBATCH --qos=regular
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/slurm-%j.out
#SBATCH --error=logs/slurm/slurm-%j.err
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=choij@cern.ch

# Optional: --pilot flag
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
echo "Extra Flags:   $EXTRA_FLAGS"
echo "=========================================="

# Load modules
echo "Loading modules..."
module load python

# Navigate to ParticleNetMD directory
cd $SLURM_SUBMIT_DIR
echo "Working directory: $(pwd)"

# Setup environment
echo "Setting up environment..."
cd ..
source setup.sh
cd ParticleNetMD

# Create logs directory
mkdir -p logs/slurm

# Run lambda sweep
echo "Starting lambda sweep..."
bash scripts/runLambdaSweep.sh $EXTRA_FLAGS

# Check exit status
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Lambda sweep completed successfully"
    echo "Completed:     $(date)"
    echo "=========================================="
else
    echo "=========================================="
    echo "ERROR: Lambda sweep failed with exit code $EXIT_CODE"
    echo "Failed:        $(date)"
    echo "=========================================="
    exit $EXIT_CODE
fi
