#!/bin/bash
#
# HTCondor wrapper script for makeBinnedTemplates DAGMan workflow
#
# Handles direct V3 steps: template, merge_template, validate, datacard,
# asymptotic, fitdiag, plots
#
# I/O Strategy for template step:
#   1. Copy input files from pnfs to local $_CONDOR_SCRATCH_DIR (via xrdcp)
#   2. Process locally
#   3. Copy output back to scratch
#
# Usage:
#   ./makeBinnedTemplates_wrapper.sh <STEP> <ERA> <CHANNEL> <MASSPOINT> <METHOD> [EXTRA_ARGS]
#
set -eo pipefail

# Parse arguments
STEP=$1      # template, merge_template, validate, datacard, asymptotic, fitdiag, plotpostfit, plotpulls, plot_score
ERA=$2
CHANNEL=$3
MASSPOINT=$4
METHOD=$5
EXTRA_ARGS="${*:6}"

# Validate required arguments
if [[ -z "$STEP" || -z "$ERA" || -z "$MASSPOINT" || -z "$METHOD" ]]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 STEP ERA CHANNEL MASSPOINT METHOD [EXTRA_ARGS]"
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
echo "Extra args: $EXTRA_ARGS"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "_CONDOR_SCRATCH_DIR: ${_CONDOR_SCRATCH_DIR:-not set}"
echo "============================================================"

# Base paths (site constants and module location come from env.sh)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
SCRATCH_WORKDIR="$SRS_REPO_DIR"
LOCAL_SCRATCH="${_CONDOR_SCRATCH_DIR:-/tmp/condor_$$}"

# pnfs paths (for xrootd access)
PNFS_BASE="$PNFS_USER_BASE/$SRS_MODULE_NAME/samples"
XROOTD_BASE="$XROOTD_USER_BASE/$SRS_MODULE_NAME/samples"

# Method directory segment: {method} or {method}_blind
METHOD_SEGMENT="$METHOD"
if [[ "$EXTRA_ARGS" == *"--blind"* ]]; then
    METHOD_SEGMENT="${METHOD}_blind"
fi

# Signal-source segment (5-segment layout). This wrapper drives the
# direct-MC chain; interp-signal jobs go through interp_templates_wrapper.sh.
SIGNAL_SOURCE="mc-signal"

# Setup environment for KNU cluster using cvmfs
setup_environment() {
    srs_setup_cmssw

    # Set up additional paths (include user site-packages for cmsstyle)
    export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$SCRATCH_WORKDIR/Common/Tools:$PYTHONPATH"
    export LD_LIBRARY_PATH="$SCRATCH_WORKDIR/Common/Tools/cpp/lib:$LD_LIBRARY_PATH"
}

# Copy one sample directory (relative to samples/) from pnfs to local
# scratch via the NFS mount (worker nodes have /pnfs/ mounted; xrootd needs
# grid auth that is unavailable there). Shared dirs hold every mass point's
# signal, so signals other than $MASSPOINT are skipped.
copy_sample_dir_to_local() {
    local relpath=$1
    local local_samples_dir=$2

    local pnfs_input_dir="$PNFS_BASE/$relpath"
    local local_input_dir="$local_samples_dir/$relpath"

    mkdir -p "$local_input_dir"

    echo "Copying input files from pnfs to local scratch..."
    echo "  Source: $pnfs_input_dir"
    echo "  Destination: $local_input_dir"

    if [[ ! -d "$pnfs_input_dir" ]]; then
        echo "ERROR: Input directory not found: $pnfs_input_dir"
        exit 1
    fi

    local f base
    for f in "$pnfs_input_dir"/*.root; do
        base=$(basename "$f")
        if [[ "$base" == MHc*_MA*.root && "$base" != "${MASSPOINT}.root" ]]; then
            continue  # another mass point's signal
        fi
        cp "$f" "$local_input_dir/"
    done

    echo "Input files copied successfully"
    ls -lh "$local_input_dir"
}

# Relative sample path (under samples/) for one era/channel of this job.
sample_relpath_for() {
    local era=$1
    local channel=$2
    if [[ "$METHOD" == "ParticleNet" ]]; then
        echo "$era/$channel/$MASSPOINT"
    elif [[ "$channel" == "SR3Mu" ]]; then
        echo "$era/SR3Mu_$(srs_pairing_variant "$MASSPOINT")"
    else
        echo "$era/$channel"
    fi
}

resolve_eras_for_request() {
    case "$ERA" in
        Run2)
            echo "2016preVFP 2016postVFP 2017 2018"
            ;;
        Run3)
            echo "2022 2022EE 2023 2023BPix"
            ;;
        All)
            echo "2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix"
            ;;
        *)
            echo "$ERA"
            ;;
    esac
}

resolve_channels_for_request() {
    case "$CHANNEL" in
        Combined)
            echo "SR1E2Mu SR3Mu"
            ;;
        *)
            echo "$CHANNEL"
            ;;
    esac
}

# Function to run template step with local scratch
run_template_local() {
    local local_workdir="$LOCAL_SCRATCH/workdir"
    mkdir -p "$local_workdir/$SRS_MODULE_NAME"

    # Create symlinks for code (read-only, from scratch)
    ln -sf "$SCRATCH_WORKDIR/$SRS_MODULE_NAME/python" "$local_workdir/$SRS_MODULE_NAME/python"
    ln -sf "$SCRATCH_WORKDIR/$SRS_MODULE_NAME/scripts" "$local_workdir/$SRS_MODULE_NAME/scripts"
    ln -sf "$SCRATCH_WORKDIR/$SRS_MODULE_NAME/configs" "$local_workdir/$SRS_MODULE_NAME/configs"
    ln -sf "$SCRATCH_WORKDIR/Common" "$local_workdir/Common"

    # Create local samples and templates directories
    mkdir -p "$local_workdir/$SRS_MODULE_NAME/samples"
    mkdir -p "$local_workdir/$SRS_MODULE_NAME/templates"

    # Copy all subera/channel inputs required by the Run-period component builder.
    read -r -a eras_to_copy <<< "$(resolve_eras_for_request)"
    read -r -a channels_to_copy <<< "$(resolve_channels_for_request)"
    for era_to_copy in "${eras_to_copy[@]}"; do
        for channel_to_copy in "${channels_to_copy[@]}"; do
            copy_sample_dir_to_local "$(sample_relpath_for "$era_to_copy" "$channel_to_copy")" \
                "$local_workdir/$SRS_MODULE_NAME/samples"
        done
    done

    # Set WORKDIR to local scratch for processing
    export WORKDIR="$local_workdir"
    cd "$local_workdir/$SRS_MODULE_NAME"
    export PATH="${PWD}/python:${PATH}"

    echo ""
    echo "Processing in local scratch: $local_workdir"
    echo "Running makeBinnedTemplates.py..."

    python3 python/makeBinnedTemplates.py \
        --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
        --method "$METHOD" $EXTRA_ARGS

    # Copy output back to scratch
    local local_output="$local_workdir/$SRS_MODULE_NAME/templates/$MASSPOINT/$METHOD_SEGMENT/$SIGNAL_SOURCE/$ERA/$CHANNEL"
    local scratch_output="$SCRATCH_WORKDIR/$SRS_MODULE_NAME/templates/$MASSPOINT/$METHOD_SEGMENT/$SIGNAL_SOURCE/$ERA/$CHANNEL"

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
    cd "$SCRATCH_WORKDIR/$SRS_MODULE_NAME"
    export PATH="${PWD}/python:${PATH}"

    case $step in
        merge_template)
            echo "Running mergeRunPeriodTemplates.py..."
            python3 python/mergeRunPeriodTemplates.py \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" $EXTRA_ARGS
            ;;
        validate)
            echo "Running validateRunPeriodTemplates.py..."
            python3 python/validateRunPeriodTemplates.py \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" $EXTRA_ARGS
            ;;
        datacard)
            echo "Running printDatacard.py..."
            python3 python/printDatacard.py \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" $EXTRA_ARGS
            ;;
        asymptotic)
            echo "Running runAsymptotic.sh..."
            bash scripts/runAsymptotic.sh \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" $EXTRA_ARGS
            ;;
        plot_score)
            # Plot ParticleNet scores (only for ParticleNet method)
            echo "Running plotParticleNetScore.py..."
            python3 python/plotParticleNetScore.py \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                $EXTRA_ARGS
            ;;
        fitdiag)
            echo "Running runFitDiagnostics.sh..."
            bash scripts/runFitDiagnostics.sh \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" $EXTRA_ARGS
            ;;
        plotpostfit)
            echo "Running plotPostfitMass.py..."
            plot_args=(
                --era "$ERA"
                --masspoint "$MASSPOINT"
                --method "$METHOD"
            )
            if [[ "$CHANNEL" != "Combined" ]]; then
                plot_args+=(--channel-scope "$CHANNEL")
            fi
            python3 python/plotPostfitMass.py "${plot_args[@]}" $EXTRA_ARGS
            ;;
        plotpulls)
            echo "Running runPullPlots.sh..."
            bash scripts/runPullPlots.sh \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" $EXTRA_ARGS
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
    merge_template|validate|datacard|asymptotic|plot_score|fitdiag|plotpostfit|plotpulls)
        # Other steps: lighter I/O, run directly on scratch
        run_on_scratch "$STEP"
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'"
        echo "Valid steps: template, merge_template, validate, datacard, asymptotic, plot_score, fitdiag, plotpostfit, plotpulls"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "============================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "============================================================"

exit $EXIT_CODE
