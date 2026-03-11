#!/bin/bash
#
# HTCondor wrapper script for makeBinnedTemplates DAGMan workflow
#
# Handles different steps: template, validate, datacard, combine_ch, asymptotic, combine_era
#
# I/O Strategy for template step:
#   1. Copy input files from pnfs to local $_CONDOR_SCRATCH_DIR (via xrdcp)
#   2. Process locally
#   3. Copy output back to scratch
#
# Usage:
#   ./makeBinnedTemplates_wrapper.sh <STEP> <ERA> <CHANNEL> <MASSPOINT> <METHOD> <BINNING> [EXTRA_ARGS]
#
set -eo pipefail

# Parse arguments
STEP=$1      # template, validate, datacard, combine_ch, asymptotic, combine_era
ERA=$2
CHANNEL=$3
MASSPOINT=$4
METHOD=$5
BINNING=$6
# Capture all remaining arguments (HTCondor may split extra_args into multiple positional params)
shift 6
EXTRA_ARGS="$*"

# Validate required arguments
if [[ -z "$STEP" || -z "$ERA" || -z "$MASSPOINT" || -z "$METHOD" || -z "$BINNING" ]]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 STEP ERA CHANNEL MASSPOINT METHOD BINNING [EXTRA_ARGS]"
    exit 1
fi

# Job info for logging
echo "============================================================"
echo "HTCondor Job: makeBinnedTemplates"
echo "Step: $STEP"
echo "Era: $ERA"
echo "Channel: $CHANNEL"
echo "Masspoint: $MASSPOINT"
echo "Method: $METHOD"
echo "Binning: $BINNING"
echo "Extra args: $EXTRA_ARGS"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "_CONDOR_SCRATCH_DIR: ${_CONDOR_SCRATCH_DIR:-not set}"
echo "============================================================"

# Base paths
SCRATCH_WORKDIR="/u/user/choij/scratch/ChargedHiggsAnalysisV3"
LOCAL_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp/condor_$$}"

# pnfs paths (for xrootd access)
PNFS_BASE="/pnfs/knu.ac.kr/data/cms/store/user/choij/SignalRegionStudyV2/samples"
XROOTD_BASE="root://cluster142.knu.ac.kr//store/user/choij/SignalRegionStudyV2/samples"

# Determine binning suffix for output path
BINNING_SUFFIX="$BINNING"
if [[ "$EXTRA_ARGS" == *"--partial-unblind"* ]]; then
    BINNING_SUFFIX="${BINNING}_partial_unblind"
elif [[ "$EXTRA_ARGS" == *"--unblind"* ]]; then
    BINNING_SUFFIX="${BINNING}_unblind"
fi

# Setup environment for KNU cluster using cvmfs
setup_environment() {
    # Source CMSSW environment from cvmfs
    source /cvmfs/cms.cern.ch/cmsset_default.sh

    # Setup CMSSW environment
    local cmssw_dir="$SCRATCH_WORKDIR/Common/CMSSW_14_1_0_pre4/src"
    if [[ -d "$cmssw_dir" ]]; then
        cd "$cmssw_dir"
        eval $(scramv1 runtime -sh)
        cd - > /dev/null
    else
        echo "ERROR: CMSSW directory not found: $cmssw_dir"
        exit 1
    fi

    # Set up additional paths (include user site-packages for cmsstyle)
    export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$SCRATCH_WORKDIR/Common/Tools:$PYTHONPATH"
    export LD_LIBRARY_PATH="$SCRATCH_WORKDIR/Common/Tools/cpp/lib:$LD_LIBRARY_PATH"
}

# Function to copy input files from pnfs to local scratch
copy_inputs_to_local() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local local_samples_dir=$4

    local pnfs_input_dir="$PNFS_BASE/$era/$channel/$masspoint"
    local xrootd_input_dir="$XROOTD_BASE/$era/$channel/$masspoint"
    local local_input_dir="$local_samples_dir/$era/$channel/$masspoint"

    mkdir -p "$local_input_dir"

    echo "Copying input files from pnfs to local scratch..."
    echo "  Source: $pnfs_input_dir"
    echo "  Destination: $local_input_dir"

    # Use cp via NFS mount (worker nodes have /pnfs/ mounted)
    # Note: xrootd requires grid authentication which is not available on worker nodes
    if [[ -d "$pnfs_input_dir" ]]; then
        echo "Using cp via NFS mount..."
        cp -v "$pnfs_input_dir"/*.root "$local_input_dir/"
    else
        echo "ERROR: Input directory not found: $pnfs_input_dir"
        exit 1
    fi

    echo "Input files copied successfully"
    ls -lh "$local_input_dir"
}

# Function to run template step with local scratch
run_template_local() {
    local local_workdir="$LOCAL_SCRATCH/workdir"
    mkdir -p "$local_workdir/SignalRegionStudyV2"

    # Create symlinks for code (read-only, from scratch)
    ln -sf "$SCRATCH_WORKDIR/SignalRegionStudyV2/python" "$local_workdir/SignalRegionStudyV2/python"
    ln -sf "$SCRATCH_WORKDIR/SignalRegionStudyV2/scripts" "$local_workdir/SignalRegionStudyV2/scripts"
    ln -sf "$SCRATCH_WORKDIR/SignalRegionStudyV2/configs" "$local_workdir/SignalRegionStudyV2/configs"
    ln -sf "$SCRATCH_WORKDIR/Common" "$local_workdir/Common"

    # Create local samples and templates directories
    mkdir -p "$local_workdir/SignalRegionStudyV2/samples"
    mkdir -p "$local_workdir/SignalRegionStudyV2/templates"

    # Copy input files from pnfs to local scratch
    copy_inputs_to_local "$ERA" "$CHANNEL" "$MASSPOINT" "$local_workdir/SignalRegionStudyV2/samples"

    # For SR3Mu, we need the SR1E2Mu fit result - link from scratch
    if [[ "$CHANNEL" == "SR3Mu" ]]; then
        local sr1e2mu_dir="$SCRATCH_WORKDIR/SignalRegionStudyV2/templates/$ERA/SR1E2Mu/$MASSPOINT/$METHOD/$BINNING_SUFFIX"
        if [[ -d "$sr1e2mu_dir" ]]; then
            mkdir -p "$local_workdir/SignalRegionStudyV2/templates/$ERA/SR1E2Mu/$MASSPOINT/$METHOD"
            ln -sf "$sr1e2mu_dir" "$local_workdir/SignalRegionStudyV2/templates/$ERA/SR1E2Mu/$MASSPOINT/$METHOD/$BINNING_SUFFIX"
            echo "Linked SR1E2Mu fit result from: $sr1e2mu_dir"
        else
            echo "ERROR: SR1E2Mu fit result not found: $sr1e2mu_dir"
            echo "SR3Mu requires SR1E2Mu to complete first."
            exit 1
        fi
    fi

    # Set WORKDIR to local scratch for processing
    export WORKDIR="$local_workdir"
    cd "$local_workdir/SignalRegionStudyV2"
    export PATH="${PWD}/python:${PATH}"

    echo ""
    echo "Processing in local scratch: $local_workdir"
    echo "Running makeBinnedTemplates.py..."

    python3 python/makeBinnedTemplates.py \
        --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
        --method "$METHOD" --binning "$BINNING" $EXTRA_ARGS

    # Copy output back to scratch
    local local_output="$local_workdir/SignalRegionStudyV2/templates/$ERA/$CHANNEL/$MASSPOINT/$METHOD/$BINNING_SUFFIX"
    local scratch_output="$SCRATCH_WORKDIR/SignalRegionStudyV2/templates/$ERA/$CHANNEL/$MASSPOINT/$METHOD/$BINNING_SUFFIX"

    if [[ -d "$local_output" ]]; then
        echo ""
        echo "Copying output to scratch: $scratch_output"
        mkdir -p "$(dirname "$scratch_output")"
        # Remove existing and copy fresh
        rm -rf "$scratch_output"
        cp -r "$local_output" "$scratch_output"
        echo "Output copied successfully"
        ls -la "$scratch_output"
    else
        echo "ERROR: Output directory not found: $local_output"
        exit 1
    fi

    # Cleanup local scratch
    echo ""
    echo "Cleaning up local scratch..."
    rm -rf "$local_workdir"
    echo "Cleanup complete"
}

# Function to run other steps directly on scratch
run_on_scratch() {
    local step=$1
    shift

    export WORKDIR="$SCRATCH_WORKDIR"
    cd "$SCRATCH_WORKDIR/SignalRegionStudyV2"
    export PATH="${PWD}/python:${PATH}"

    case $step in
        validate)
            echo "Running checkTemplates.py..."
            python3 python/checkTemplates.py \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" --binning "$BINNING" $EXTRA_ARGS
            ;;
        datacard)
            echo "Running printDatacard.py..."
            python3 python/printDatacard.py \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" --binning "$BINNING" $EXTRA_ARGS
            ;;
        combine_ch)
            echo "Running combineDatacards.py (channel combination)..."
            python3 python/combineDatacards.py --mode channel \
                --era "$ERA" --masspoint "$MASSPOINT" \
                --method "$METHOD" --binning "$BINNING" $EXTRA_ARGS
            ;;
        asymptotic)
            echo "Running runAsymptotic.sh..."
            bash scripts/runAsymptotic.sh \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" --binning "$BINNING" $EXTRA_ARGS
            ;;
        combine_era)
            # $ERA contains comma-separated list of eras
            # $CHANNEL contains output era name (Run2, Run3, All)
            echo "Running combineDatacards.py (era combination)..."
            python3 python/combineDatacards.py --mode era \
                --eras "$ERA" --channel Combined --masspoint "$MASSPOINT" \
                --method "$METHOD" --binning "$BINNING" --output-era "$CHANNEL" $EXTRA_ARGS
            ;;
        plot_score)
            # Plot ParticleNet scores (only for ParticleNet method)
            echo "Running plotParticleNetScore.py..."
            python3 python/plotParticleNetScore.py \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --binning "$BINNING" $EXTRA_ARGS
            ;;
        fitdiag)
            echo "Running runFitDiagnostics.sh..."
            bash scripts/runFitDiagnostics.sh \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" --binning "$BINNING" $EXTRA_ARGS
            ;;
        plotpostfit)
            echo "Running plotPostfit.py..."
            python3 python/plotPostfit.py \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" --binning "$BINNING" $EXTRA_ARGS
            ;;
        *)
            echo "ERROR: Unknown step '$step'"
            exit 1
            ;;
    esac
}

# Main execution
setup_environment

case $STEP in
    template)
        # Template generation: copy inputs from pnfs, process locally, copy output back
        run_template_local
        ;;
    validate|datacard|combine_ch|asymptotic|combine_era|plot_score|fitdiag|plotpostfit)
        # Other steps: lighter I/O, run directly on scratch
        run_on_scratch "$STEP"
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'"
        echo "Valid steps: template, validate, datacard, combine_ch, asymptotic, combine_era, plot_score, fitdiag, plotpostfit"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "============================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "============================================================"

exit $EXIT_CODE
