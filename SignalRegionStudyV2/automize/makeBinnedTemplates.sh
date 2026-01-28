#!/bin/bash
set -euo pipefail

# Get the script directory (SignalRegionStudyV2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Mass points
MASSPOINTs_BASELINE=(
    "MHc70_MA15" "MHc70_MA18" "MHc70_MA40" "MHc70_MA55" "MHc70_MA65"
    "MHc85_MA15" "MHc85_MA70" "MHc85_MA80"      # 85_21 is missng in 16a
    "MHc100_MA15" "MHc100_MA24" "MHc100_MA60" "MHc100_MA75" "MHc100_MA95"
    "MHc115_MA15" "MHc115_MA27" "MHc115_MA87" "MHc115_MA110"
    "MHc130_MA15" "MHc130_MA30" "MHc130_MA55" "MHc130_MA83" "MHc130_MA90" "MHc130_MA100" "MHc130_MA125"
    "MHc145_MA15" "MHc145_MA35" "MHc145_MA92" "MHc145_MA140"
    "MHc160_MA15" "MHc160_MA50" "MHc160_MA85" "MHc160_MA98" "MHc160_MA120" "MHc160_MA135" "MHc160_MA155"
)
MASSPOINTs_PARTICLENET=(
    "MHc100_MA95" "MHc130_MA90" "MHc160_MA85" "MHc115_MA87" "MHc145_MA92" "MHc160_MA98"
)

# Eras
ERAs_RUN2=("2016preVFP" "2016postVFP" "2017" "2018")
ERAs_RUN3=("2022" "2022EE" "2023" "2023BPix")

# Channels - SR1E2Mu must be processed first (SR3Mu loads fit from SR1E2Mu)
CHANNELs=("SR1E2Mu" "SR3Mu")

# HTCondor settings
CONDOR_DIR="$SCRIPT_DIR/condor"

# Export PATH to include python scripts
export PATH="${PWD}/python:${PATH}"

# Parse command line arguments
MODE="all"  # Options: all, run2, run3
SINGLE_ERA=""  # Single era mode (overrides MODE)
METHOD="Baseline"  # Options: Baseline, ParticleNet
EXTRA_ARGS=""
BINNING="extended"
# DO_PLOT_SCORE is set after METHOD is parsed (default depends on method)
DO_PLOT_SCORE_SET=false
# Datacard and limits options (enabled by default)
DO_PRINT_DATACARD=true
DO_COMBINE_DATACARDS=true
DO_RUN_ASYMPTOTIC=true
# Dry run mode
DRY_RUN=false
# Start-from step (template, datacard, validate, combine, asymptotic, combine_era, asymptotic_combined)
START_FROM="template"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="${2,,}"  # Convert to lowercase
            shift 2
            ;;
        --era)
            SINGLE_ERA="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --binning)
            BINNING="$2"
            EXTRA_ARGS="$EXTRA_ARGS --binning $2"
            shift 2
            ;;
        --unblind)
            EXTRA_ARGS="$EXTRA_ARGS --unblind"
            shift
            ;;
        --partial-unblind)
            EXTRA_ARGS="$EXTRA_ARGS --partial-unblind"
            shift
            ;;
        --debug)
            EXTRA_ARGS="$EXTRA_ARGS --debug"
            shift
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --plot-score)
            DO_PLOT_SCORE_SET=true
            DO_PLOT_SCORE=true
            shift
            ;;
        --no-plot-score)
            DO_PLOT_SCORE_SET=true
            DO_PLOT_SCORE=false
            shift
            ;;
        --validationOnly)
            START_FROM="datacard"
            shift
            ;;
        --printDatacard)
            DO_PRINT_DATACARD=true
            shift
            ;;
        --no-printDatacard)
            DO_PRINT_DATACARD=false
            shift
            ;;
        --combineDatacards)
            DO_COMBINE_DATACARDS=true
            shift
            ;;
        --no-combineDatacards)
            DO_COMBINE_DATACARDS=false
            shift
            ;;
        --runAsymptotic)
            DO_RUN_ASYMPTOTIC=true
            shift
            ;;
        --no-runAsymptotic)
            DO_RUN_ASYMPTOTIC=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --condor)
            echo "NOTE: --condor is now the default (and only) execution mode. Flag ignored."
            shift
            ;;
        --help)
            echo "Usage: $0 [--mode <all|run2|run3>] [--method <Baseline|ParticleNet>] [--binning <extended|uniform>] [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  all    - Process Run2 and Run3"
            echo "  run2   - Process Run2 only"
            echo "  run3   - Process Run3 only"
            echo ""
            echo "Single Era:"
            echo "  --era <era>  - Process single era only (e.g., --era 2018)"
            echo "                 Valid: 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix"
            echo ""
            echo "Methods:"
            echo "  Baseline    - Standard cut-based analysis"
            echo "  ParticleNet - MVA-based analysis (score plotting enabled by default)"
            echo ""
            echo "Template Options:"
            echo "  --binning extended   - Use extended binning (default, 19 bins)"
            echo "  --binning uniform    - Use uniform binning (15 bins)"
            echo "  --unblind            - Use real data for data_obs"
            echo "  --partial-unblind    - Unblind low score region (requires ParticleNet)"
            echo "  --validationOnly     - Skip template generation (sets --start-from datacard)"
            echo "  --plot-score         - Enable ParticleNet score plotting (default for ParticleNet)"
            echo "  --no-plot-score      - Disable ParticleNet score plotting"
            echo "  --debug              - Enable debug logging"
            echo ""
            echo "Workflow Control:"
            echo "  --start-from <step>     - Skip already-completed steps (marks them as DONE in DAG)"
            echo "                            Values: template (default), datacard, validate, combine,"
            echo "                                    asymptotic, combine_era, asymptotic_combined"
            echo ""
            echo "Datacard & Limits Options (enabled by default):"
            echo "  --no-printDatacard      - Skip datacard generation"
            echo "  --no-combineDatacards   - Skip datacard combination"
            echo "  --no-runAsymptotic      - Skip asymptotic limits"
            echo ""
            echo "HTCondor Options:"
            echo "  --dry-run               - Generate DAG without submitting"
            echo "  --condor                - (No-op, condor is now the only execution mode)"
            echo ""
            echo "Examples:"
            echo "  # Full pipeline (default): templates + datacards + combination + limits"
            echo "  $0 --mode all --method Baseline --binning extended"
            echo ""
            echo "  # Start from era combination (per-era work already done)"
            echo "  $0 --mode all --binning extended --start-from combine_era"
            echo ""
            echo "  # Only run final asymptotic"
            echo "  $0 --mode all --binning extended --start-from asymptotic_combined"
            echo ""
            echo "  # Dry run to inspect DAG"
            echo "  $0 --mode all --binning extended --start-from combine_era --dry-run"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate START_FROM value
case "$START_FROM" in
    template|datacard|validate|combine|asymptotic|combine_era|asymptotic_combined) ;;
    *)
        echo "ERROR: Invalid --start-from value '$START_FROM'"
        echo "Valid values: template, datacard, validate, combine, asymptotic, combine_era, asymptotic_combined"
        exit 1
        ;;
esac

# Set defaults: plot-score enabled for ParticleNet only (if not explicitly set)
if [[ "$DO_PLOT_SCORE_SET" == "false" ]]; then
    if [[ "$METHOD" == "ParticleNet" ]]; then
        DO_PLOT_SCORE=true
    else
        DO_PLOT_SCORE=false
    fi
fi

# Select mass points based on method
if [[ "$METHOD" == "ParticleNet" ]]; then
    MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}")
else
    MASSPOINTs=("${MASSPOINTs_BASELINE[@]}")
fi

echo "============================================================"
echo "SignalRegionStudyV2 Template Generation"
if [[ -n "$SINGLE_ERA" ]]; then
    echo "Era: $SINGLE_ERA (single era mode)"
else
    echo "Mode: $MODE"
fi
echo "Method: $METHOD"
echo "Mass points: ${MASSPOINTs[*]}"
echo "Binning: $BINNING"
echo "Plot score: $DO_PLOT_SCORE"
echo "Start from: $START_FROM"
echo "Print datacard: $DO_PRINT_DATACARD"
echo "Combine datacards: $DO_COMBINE_DATACARDS"
echo "Run asymptotic: $DO_RUN_ASYMPTOTIC"
echo "Execution: HTCondor DAGMan (dry-run: $DRY_RUN)"
echo "Extra args: $EXTRA_ARGS"
echo "============================================================"

# =============================================================================
# Helper Functions
# =============================================================================

# Function to get the template directory path (absolute)
function get_template_dir() {
    local era=$1 channel=$2 masspoint=$3 method=$4 binning=$5 extra_args=${6:-}
    local binning_suffix="$binning"
    if [[ "$extra_args" == *"--partial-unblind"* ]]; then
        binning_suffix="${binning}_partial_unblind"
    elif [[ "$extra_args" == *"--unblind"* ]]; then
        binning_suffix="${binning}_unblind"
    fi
    echo "${SCRIPT_DIR}/templates/${era}/${channel}/${masspoint}/${method}/${binning_suffix}"
}

# Function to generate templates for a single configuration
function generate_template() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local method=$4
    local binning=$5
    local extra_args=${6:-}

    local cmd="python3 python/makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method $method $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"

    echo "Generating template: era=$era, channel=$channel, masspoint=$masspoint, method=$method"
    # Run python script and capture output (logs dir created AFTER since python script may recreate template dir)
    local output
    output=$(python3 python/makeBinnedTemplates.py --era "$era" --channel "$channel" --masspoint "$masspoint" --method "$method" $extra_args 2>&1)
    local exit_code=$?

    # Create logs directory after python script (which may have recreated the template dir)
    mkdir -p "$log_dir"
    echo "$output" > "${log_dir}/makeBinnedTemplates.log"
    echo "$output"

    return $exit_code
}

# Function to validate templates for a single configuration
function validate_template() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local method=$4
    local binning=$5
    local extra_args=${6:-}

    local cmd="python3 python/checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method $method --binning $binning $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Validating template: era=$era, channel=$channel, masspoint=$masspoint"
    python3 python/checkTemplates.py --era "$era" --channel "$channel" --masspoint "$masspoint" --method "$method" --binning "$binning" $extra_args \
        2>&1 | tee "${log_dir}/checkTemplates.log"
}

# Function to plot ParticleNet scores for a single configuration
function plot_score() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local binning=$4
    local extra_args=${5:-}

    local cmd="python3 python/plotParticleNetScore.py --era $era --channel $channel --masspoint $masspoint --binning $binning $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$METHOD" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Plotting ParticleNet score: era=$era, channel=$channel, masspoint=$masspoint"
    python3 python/plotParticleNetScore.py --era "$era" --channel "$channel" --masspoint "$masspoint" --binning "$binning" $extra_args \
        2>&1 | tee "${log_dir}/plotParticleNetScore.log"
}

# Function to generate datacards for a single configuration
function print_datacard() {
    local era=$1 channel=$2 masspoint=$3 method=$4 binning=$5 extra_args=${6:-}

    local cmd="python3 python/printDatacard.py --era $era --channel $channel --masspoint $masspoint --method $method --binning $binning $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Generating datacard: era=$era, channel=$channel, masspoint=$masspoint"
    python3 python/printDatacard.py --era "$era" --channel "$channel" \
        --masspoint "$masspoint" --method "$method" --binning "$binning" $extra_args \
        2>&1 | tee "${log_dir}/printDatacard.log"
}

# Function to run asymptotic limits for a single configuration
function run_asymptotic() {
    local era=$1 channel=$2 masspoint=$3 method=$4 binning=$5

    local cmd="bash scripts/runAsymptotic.sh --era $era --channel $channel --masspoint $masspoint --method $method --binning $binning"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "$channel" "$masspoint" "$method" "$binning" "")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Running asymptotic: era=$era, channel=$channel, masspoint=$masspoint"
    bash scripts/runAsymptotic.sh --era "$era" --channel "$channel" \
        --masspoint "$masspoint" --method "$method" --binning "$binning" \
        2>&1 | tee "${log_dir}/runAsymptotic.log"
}

# Function to combine channels (SR1E2Mu + SR3Mu -> Combined)
function combine_channels() {
    local era=$1 masspoint=$2 method=$3 binning=$4 extra_args=${5:-}

    local cmd="python3 python/combineDatacards.py --mode channel --era $era --masspoint $masspoint --method $method --binning $binning $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$era" "Combined" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Combining channels: era=$era, masspoint=$masspoint"
    python3 python/combineDatacards.py --mode channel \
        --era "$era" --masspoint "$masspoint" --method "$method" --binning "$binning" $extra_args \
        2>&1 | tee "${log_dir}/combineDatacards_channels.log"
}

# Function to combine eras
function combine_eras() {
    local eras=$1 channel=$2 masspoint=$3 method=$4 binning=$5 output_era=$6 extra_args=${7:-}

    local cmd="python3 python/combineDatacards.py --mode era --eras $eras --channel $channel --masspoint $masspoint --method $method --binning $binning --output-era $output_era $extra_args"

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $cmd"
        return 0
    fi

    local template_dir
    template_dir=$(get_template_dir "$output_era" "$channel" "$masspoint" "$method" "$binning" "$extra_args")
    local log_dir="${template_dir}/logs"
    mkdir -p "$log_dir"

    echo "Combining eras: eras=$eras -> $output_era, channel=$channel, masspoint=$masspoint"
    python3 python/combineDatacards.py --mode era \
        --eras "$eras" --channel "$channel" --masspoint "$masspoint" \
        --method "$method" --binning "$binning" --output-era "$output_era" $extra_args \
        2>&1 | tee "${log_dir}/combineDatacards_eras.log"
}

# Determine extra args for various steps
VALIDATE_EXTRA_ARGS=""
DATACARD_EXTRA_ARGS=""
COMBINE_EXTRA_ARGS=""
PLOT_EXTRA_ARGS=""
if [[ "$EXTRA_ARGS" == *"--unblind"* ]]; then
    VALIDATE_EXTRA_ARGS="--unblind"
    DATACARD_EXTRA_ARGS="--unblind"
    PLOT_EXTRA_ARGS="--unblind"
elif [[ "$EXTRA_ARGS" == *"--partial-unblind"* ]]; then
    VALIDATE_EXTRA_ARGS="--partial-unblind"
    DATACARD_EXTRA_ARGS="--partial-unblind"
    COMBINE_EXTRA_ARGS="--partial-unblind"
    PLOT_EXTRA_ARGS="--partial-unblind"
fi

# =============================================================================
# HTCondor Submission Function
# =============================================================================

# Function to submit HTCondor jobs for template generation
function submit_condor_jobs() {
    local -n eras_ref=$1
    local -n masspoints_ref=$2
    local method=$3
    local binning=$4
    local extra_args=${5:-}

    # Create timestamped job directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$CONDOR_DIR/jobs_templates_${timestamp}"
    mkdir -p "$job_dir/logs"

    # Generate job_params.txt with all era/channel/masspoint combinations
    # Process SR1E2Mu first (required for SR3Mu fit)
    local params_file="$job_dir/job_params.txt"
    > "$params_file"

    # SR1E2Mu jobs first
    for era in "${eras_ref[@]}"; do
        for masspoint in "${masspoints_ref[@]}"; do
            echo "$era,SR1E2Mu,$masspoint,$method,$binning,$extra_args" >> "$params_file"
        done
    done

    # SR3Mu jobs second
    for era in "${eras_ref[@]}"; do
        for masspoint in "${masspoints_ref[@]}"; do
            echo "$era,SR3Mu,$masspoint,$method,$binning,$extra_args" >> "$params_file"
        done
    done

    local num_jobs=$(wc -l < "$params_file")
    echo "Generated $num_jobs jobs in $params_file"

    # Generate HTCondor submission file
    local sub_file="$job_dir/makeBinnedTemplates.sub"
    cat > "$sub_file" << 'CONDOR_SUB'
JobBatchName = makeBinnedTemplates
universe = vanilla
executable = WRAPPER_PATH
arguments = $(era) $(channel) $(masspoint) $(method) $(binning) $(extra_args)
output = logs/template_$(era)_$(channel)_$(masspoint).out
error = logs/template_$(era)_$(channel)_$(masspoint).err
log = makeBinnedTemplates.log

request_cpus = 1
request_memory = 2GB
request_disk = 2GB

getenv = True
should_transfer_files = NO

queue era,channel,masspoint,method,binning,extra_args from job_params.txt
CONDOR_SUB

    # Replace WRAPPER_PATH with actual path
    sed -i "s|WRAPPER_PATH|$CONDOR_DIR/makeBinnedTemplates_wrapper.sh|g" "$sub_file"

    echo "Created submission file: $sub_file"
    echo ""
    echo "NOTE: SR3Mu depends on SR1E2Mu templates for fit. Jobs are ordered but not enforced."
    echo "      Consider submitting SR1E2Mu first, wait for completion, then submit SR3Mu."
    echo ""

    # Submit or dry-run
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would submit: condor_submit $sub_file"
        echo "[DRY-RUN] Job directory: $job_dir"
        echo "[DRY-RUN] First 10 jobs:"
        head -10 "$params_file" | while IFS=, read -r era channel masspoint method binning extra; do
            echo "  era=$era channel=$channel masspoint=$masspoint method=$method binning=$binning"
        done
    else
        echo "Submitting to HTCondor..."
        cd "$job_dir"
        condor_submit makeBinnedTemplates.sub
        cd - > /dev/null
    fi
}

# =============================================================================
# DAGMan Workflow Functions
# =============================================================================

# Function to generate a DAG file for a single masspoint
# This creates the full workflow with proper dependencies
function generate_dag_file() {
    local masspoint=$1
    local mode=$2          # "run2", "run3", "all", "single_run2", or "single_run3"
    local method=$3
    local binning=$4
    local extra_args=${5:-}
    local dag_file=$6
    local single_era=${7:-}  # Optional: specific era for single era mode
    local start_from=${8:-template}  # Step to start from

    # Define eras based on mode
    local -a run2_eras=("2016preVFP" "2016postVFP" "2017" "2018")
    local -a run3_eras=("2022" "2022EE" "2023" "2023BPix")
    local -a single_era_array=()

    # For single era mode, override the era arrays
    if [[ -n "$single_era" ]]; then
        single_era_array=("$single_era")
    fi

    # Step level mapping for --start-from
    # template=0, datacard=1, validate=2, combine=3, asymptotic=4, combine_era=5, asymptotic_combined=6
    step_to_level() {
        case "$1" in
            template)             echo 0 ;;
            datacard)             echo 1 ;;
            validate)             echo 2 ;;
            combine|combine_ch)   echo 3 ;;
            asymptotic)           echo 4 ;;
            combine_era)          echo 5 ;;
            asymptotic_combined)  echo 6 ;;
            *)                    echo 0 ;;
        esac
    }

    local start_level
    start_level=$(step_to_level "$start_from")

    # Determine DONE suffix for a given step type
    # A job is marked DONE if:
    #   - Its step level < start_level (already completed), OR
    #   - Its step is disabled by --no-* flags
    job_done_suffix() {
        local step_type=$1
        local level
        level=$(step_to_level "$step_type")

        # Check --start-from
        if [[ $level -lt $start_level ]]; then
            echo " DONE"
            return
        fi

        # Check --no-* flags
        case "$step_type" in
            datacard)
                if [[ "$DO_PRINT_DATACARD" == "false" ]]; then echo " DONE"; return; fi
                ;;
            combine_ch|combine_era)
                if [[ "$DO_COMBINE_DATACARDS" == "false" ]]; then echo " DONE"; return; fi
                ;;
            asymptotic|asymptotic_combined)
                if [[ "$DO_RUN_ASYMPTOTIC" == "false" ]]; then echo " DONE"; return; fi
                ;;
        esac

        echo ""
    }

    # Start DAG file
    cat > "$dag_file" << EOF
# DAG for $masspoint (mode: $mode, start-from: $start_from)
CONFIG dagman.config

EOF

    # Helper function to add jobs for a run period
    add_run_period_jobs() {
        local run_name=$1  # "Run2" or "Run3"
        local -n eras=$2

        # Step 1: SR1E2Mu Templates
        local done_sfx
        done_sfx=$(job_done_suffix "template")
        for era in "${eras[@]}"; do
            echo "JOB template_SR1E2Mu_${era} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS template_SR1E2Mu_${era} step=\"template\" era=\"${era}\" channel=\"SR1E2Mu\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"${extra_args}\"" >> "$dag_file"
        done

        # Step 2: SR3Mu Templates
        for era in "${eras[@]}"; do
            echo "JOB template_SR3Mu_${era} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS template_SR3Mu_${era} step=\"template\" era=\"${era}\" channel=\"SR3Mu\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"${extra_args}\"" >> "$dag_file"
        done

        # Step 3: Datacard (produces lowstat.json + filtered shapes.root)
        done_sfx=$(job_done_suffix "datacard")
        for era in "${eras[@]}"; do
            for channel in SR1E2Mu SR3Mu; do
                echo "JOB datacard_${channel}_${era} jobs.sub${done_sfx}" >> "$dag_file"
                echo "VARS datacard_${channel}_${era} step=\"datacard\" era=\"${era}\" channel=\"${channel}\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"${extra_args}\"" >> "$dag_file"
            done
        done

        # Step 4: Validate (uses lowstat.json from datacard step)
        done_sfx=$(job_done_suffix "validate")
        for era in "${eras[@]}"; do
            for channel in SR1E2Mu SR3Mu; do
                echo "JOB validate_${channel}_${era} jobs.sub${done_sfx}" >> "$dag_file"
                echo "VARS validate_${channel}_${era} step=\"validate\" era=\"${era}\" channel=\"${channel}\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"${extra_args}\"" >> "$dag_file"
            done
        done

        # Step 5: Combine channels
        done_sfx=$(job_done_suffix "combine_ch")
        for era in "${eras[@]}"; do
            echo "JOB combine_ch_${era} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS combine_ch_${era} step=\"combine_ch\" era=\"${era}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"${extra_args}\"" >> "$dag_file"
        done

        # Step 6: Per-era Asymptotic
        done_sfx=$(job_done_suffix "asymptotic")
        for era in "${eras[@]}"; do
            echo "JOB asymptotic_${era} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS asymptotic_${era} step=\"asymptotic\" era=\"${era}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"\"" >> "$dag_file"
        done

        # Step 7: Combine eras
        done_sfx=$(job_done_suffix "combine_era")
        local eras_csv=$(IFS=,; echo "${eras[*]}")
        echo "JOB combine_era_${run_name} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS combine_era_${run_name} step=\"combine_era\" era=\"${eras_csv}\" channel=\"${run_name}\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"${extra_args}\"" >> "$dag_file"

        # Step 8: Combined Asymptotic (run-period level)
        done_sfx=$(job_done_suffix "asymptotic_combined")
        echo "JOB asymptotic_${run_name} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS asymptotic_${run_name} step=\"asymptotic\" era=\"${run_name}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"\"" >> "$dag_file"
    }

    # Helper function for single era mode (no era combination step)
    add_single_era_jobs() {
        local era=$1
        local mp=$2
        local meth=$3
        local bin=$4
        local extra=${5:-}

        # Step 1: SR1E2Mu Template
        local done_sfx
        done_sfx=$(job_done_suffix "template")
        echo "JOB template_SR1E2Mu_${era} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS template_SR1E2Mu_${era} step=\"template\" era=\"${era}\" channel=\"SR1E2Mu\" masspoint=\"${mp}\" method=\"${meth}\" binning=\"${bin}\" extra_args=\"${extra}\"" >> "$dag_file"

        # Step 2: SR3Mu Template
        echo "JOB template_SR3Mu_${era} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS template_SR3Mu_${era} step=\"template\" era=\"${era}\" channel=\"SR3Mu\" masspoint=\"${mp}\" method=\"${meth}\" binning=\"${bin}\" extra_args=\"${extra}\"" >> "$dag_file"

        # Step 3: Datacard (produces lowstat.json + filtered shapes.root)
        done_sfx=$(job_done_suffix "datacard")
        for channel in SR1E2Mu SR3Mu; do
            echo "JOB datacard_${channel}_${era} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS datacard_${channel}_${era} step=\"datacard\" era=\"${era}\" channel=\"${channel}\" masspoint=\"${mp}\" method=\"${meth}\" binning=\"${bin}\" extra_args=\"${extra}\"" >> "$dag_file"
        done

        # Step 4: Validate (uses lowstat.json from datacard step)
        done_sfx=$(job_done_suffix "validate")
        for channel in SR1E2Mu SR3Mu; do
            echo "JOB validate_${channel}_${era} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS validate_${channel}_${era} step=\"validate\" era=\"${era}\" channel=\"${channel}\" masspoint=\"${mp}\" method=\"${meth}\" binning=\"${bin}\" extra_args=\"${extra}\"" >> "$dag_file"
        done

        # Step 5: Combine channels
        done_sfx=$(job_done_suffix "combine_ch")
        echo "JOB combine_ch_${era} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS combine_ch_${era} step=\"combine_ch\" era=\"${era}\" channel=\"Combined\" masspoint=\"${mp}\" method=\"${meth}\" binning=\"${bin}\" extra_args=\"${extra}\"" >> "$dag_file"

        # Step 6: Asymptotic (final step for single era)
        done_sfx=$(job_done_suffix "asymptotic")
        echo "JOB asymptotic_${era} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS asymptotic_${era} step=\"asymptotic\" era=\"${era}\" channel=\"Combined\" masspoint=\"${mp}\" method=\"${meth}\" binning=\"${bin}\" extra_args=\"\"" >> "$dag_file"
    }

    # Add jobs based on mode
    if [[ "$mode" == "single_run2" || "$mode" == "single_run3" ]]; then
        # Single era mode: simplified workflow (no era combination)
        add_single_era_jobs "$single_era" "$masspoint" "$method" "$binning" "$extra_args"
    else
        # Multi-era modes
        if [[ "$mode" == "run2" || "$mode" == "all" ]]; then
            add_run_period_jobs "Run2" run2_eras
        fi
        if [[ "$mode" == "run3" || "$mode" == "all" ]]; then
            add_run_period_jobs "Run3" run3_eras
        fi

        # For --mode all: add final combination
        if [[ "$mode" == "all" ]]; then
            local done_sfx
            done_sfx=$(job_done_suffix "combine_era")
            echo "JOB combine_era_All jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS combine_era_All step=\"combine_era\" era=\"Run2,Run3\" channel=\"All\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"${extra_args}\"" >> "$dag_file"
            done_sfx=$(job_done_suffix "asymptotic_combined")
            echo "JOB asymptotic_All jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS asymptotic_All step=\"asymptotic\" era=\"All\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" extra_args=\"\"" >> "$dag_file"
        fi
    fi

    # Add dependencies
    echo "" >> "$dag_file"
    echo "# Dependencies" >> "$dag_file"

    add_run_period_deps() {
        local run_name=$1
        local -n eras=$2

        # SR1E2Mu templates -> SR3Mu templates (SR3Mu depends on SR1E2Mu fit)
        local sr1e2mu_jobs=$(printf "template_SR1E2Mu_%s " "${eras[@]}")
        local sr3mu_jobs=$(printf "template_SR3Mu_%s " "${eras[@]}")
        echo "PARENT $sr1e2mu_jobs CHILD $sr3mu_jobs" >> "$dag_file"

        # SR3Mu templates -> Datacard (produces lowstat.json + filtered shapes.root)
        local datacard_jobs=""
        for era in "${eras[@]}"; do
            datacard_jobs+="datacard_SR1E2Mu_${era} datacard_SR3Mu_${era} "
        done
        echo "PARENT $sr3mu_jobs CHILD $datacard_jobs" >> "$dag_file"

        # Datacard -> Validate (needs lowstat.json from datacard step)
        local validate_jobs=""
        for era in "${eras[@]}"; do
            validate_jobs+="validate_SR1E2Mu_${era} validate_SR3Mu_${era} "
        done
        echo "PARENT $datacard_jobs CHILD $validate_jobs" >> "$dag_file"

        # Validate -> Combine channels
        local combine_ch_jobs=$(printf "combine_ch_%s " "${eras[@]}")
        echo "PARENT $validate_jobs CHILD $combine_ch_jobs" >> "$dag_file"

        # Combine channels -> Per-era Asymptotic
        local asymptotic_jobs=$(printf "asymptotic_%s " "${eras[@]}")
        echo "PARENT $combine_ch_jobs CHILD $asymptotic_jobs" >> "$dag_file"

        # Per-era Asymptotic -> Era combination
        echo "PARENT $asymptotic_jobs CHILD combine_era_${run_name}" >> "$dag_file"

        # Era combination -> Combined Asymptotic
        echo "PARENT combine_era_${run_name} CHILD asymptotic_${run_name}" >> "$dag_file"
    }

    # Helper function for single era dependencies (no era combination)
    add_single_era_deps() {
        local era=$1

        # SR1E2Mu template -> SR3Mu template
        echo "PARENT template_SR1E2Mu_${era} CHILD template_SR3Mu_${era}" >> "$dag_file"

        # SR3Mu template -> Datacard (produces lowstat.json + filtered shapes.root)
        echo "PARENT template_SR3Mu_${era} CHILD datacard_SR1E2Mu_${era} datacard_SR3Mu_${era}" >> "$dag_file"

        # Datacard -> Validate (needs lowstat.json from datacard step)
        echo "PARENT datacard_SR1E2Mu_${era} datacard_SR3Mu_${era} CHILD validate_SR1E2Mu_${era} validate_SR3Mu_${era}" >> "$dag_file"

        # Validate -> Combine channels
        echo "PARENT validate_SR1E2Mu_${era} validate_SR3Mu_${era} CHILD combine_ch_${era}" >> "$dag_file"

        # Combine channels -> Asymptotic
        echo "PARENT combine_ch_${era} CHILD asymptotic_${era}" >> "$dag_file"
    }

    if [[ "$mode" == "single_run2" || "$mode" == "single_run3" ]]; then
        add_single_era_deps "$single_era"
    else
        if [[ "$mode" == "run2" || "$mode" == "all" ]]; then
            add_run_period_deps "Run2" run2_eras
        fi
        if [[ "$mode" == "run3" || "$mode" == "all" ]]; then
            add_run_period_deps "Run3" run3_eras
        fi

        # Final combination dependencies for --mode all
        if [[ "$mode" == "all" ]]; then
            echo "PARENT asymptotic_Run2 asymptotic_Run3 CHILD combine_era_All" >> "$dag_file"
            echo "PARENT combine_era_All CHILD asymptotic_All" >> "$dag_file"
        fi
    fi
}

# Function to submit DAGMan workflows for all masspoints
function submit_condor_dags() {
    local mode=$1          # "run2", "run3", "all", "single_run2", or "single_run3"
    local -n masspoints_ref=$2
    local method=$3
    local binning=$4
    local extra_args=${5:-}
    local single_era=${6:-}  # Optional: specific era for single era mode

    # Create timestamped job directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local dir_suffix="$mode"
    if [[ -n "$single_era" ]]; then
        dir_suffix="${single_era}"
    fi
    local job_dir="$CONDOR_DIR/jobs_dag_${dir_suffix}_${timestamp}"
    mkdir -p "$job_dir"

    # Copy dagman.config to job directory
    cp "$CONDOR_DIR/dagman.config" "$job_dir/"

    # Generate DAG for each masspoint
    for masspoint in "${masspoints_ref[@]}"; do
        local mp_dir="$job_dir/$masspoint"
        mkdir -p "$mp_dir/logs"

        # Create jobs.sub for this masspoint
        cat > "$mp_dir/jobs.sub" << EOF
JobBatchName = ${masspoint}
universe = vanilla
executable = $CONDOR_DIR/makeBinnedTemplates_wrapper.sh
arguments = \$(step) \$(era) \$(channel) \$(masspoint) \$(method) \$(binning) \$(extra_args)
output = logs/\$(step)_\$(channel)_\$(era).out
error = logs/\$(step)_\$(channel)_\$(era).err
log = dag.log
request_cpus = 1
request_memory = 2GB
request_disk = 2GB
getenv = True
should_transfer_files = NO
queue
EOF

        # Copy dagman.config to masspoint directory
        cp "$job_dir/dagman.config" "$mp_dir/"

        # Generate DAG file
        generate_dag_file "$masspoint" "$mode" "$method" "$binning" "$extra_args" "$mp_dir/dag.dag" "$single_era" "$START_FROM"

        echo "Generated DAG: $mp_dir/dag.dag"
    done

    # Create submit_all.sh script
    cat > "$job_dir/submit_all.sh" << 'EOF'
#!/bin/bash
set -e
for mp_dir in */; do
    if [[ -f "$mp_dir/dag.dag" ]]; then
        echo "Submitting DAG for ${mp_dir%/}..."
        (cd "$mp_dir" && condor_submit_dag dag.dag)
    fi
done
echo "All DAGs submitted!"
EOF
    chmod +x "$job_dir/submit_all.sh"

    # Create status_all.sh script for monitoring
    cat > "$job_dir/status_all.sh" << 'EOF'
#!/bin/bash
echo "DAG Status Summary:"
echo "==================="
for mp_dir in */; do
    if [[ -f "$mp_dir/dag.dag" ]]; then
        mp_name="${mp_dir%/}"
        if [[ -f "$mp_dir/dag.dag.dagman.out" ]]; then
            # Extract status from dagman output
            done=$(grep -c "ULOG_JOB_TERMINATED" "$mp_dir/dag.dag.dagman.out" 2>/dev/null || echo 0)
            total=$(grep -c "^JOB " "$mp_dir/dag.dag" 2>/dev/null || echo 0)
            echo "$mp_name: $done/$total jobs completed"
        else
            echo "$mp_name: not started"
        fi
    fi
done
EOF
    chmod +x "$job_dir/status_all.sh"

    echo ""
    echo "========================================"
    echo "Generated DAGMan workflows in: $job_dir"
    if [[ -n "$single_era" ]]; then
        echo "Era: $single_era (single era mode)"
    else
        echo "Mode: $mode"
    fi
    echo "Total masspoints: ${#masspoints_ref[@]}"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] To submit all DAGs:"
        echo "  cd $job_dir && ./submit_all.sh"
        echo ""
        echo "[DRY-RUN] To check status:"
        echo "  cd $job_dir && ./status_all.sh"
        echo "  condor_q -dag"
    else
        echo "Submitting all DAGs..."
        cd "$job_dir" && ./submit_all.sh
        cd - > /dev/null
        echo ""
        echo "Monitor with:"
        echo "  condor_q -dag"
        echo "  cd $job_dir && ./status_all.sh"
    fi
}

# =============================================================================
# Main Execution
# =============================================================================

echo ""
echo "============================================================"
echo "Generating DAGMan workflows for HTCondor..."
echo "============================================================"

# Determine effective mode for DAG generation
EFFECTIVE_MODE="$MODE"
if [[ -n "$SINGLE_ERA" ]]; then
    # Single era mode: determine if run2 or run3
    if [[ " ${ERAs_RUN2[*]} " =~ " ${SINGLE_ERA} " ]]; then
        EFFECTIVE_MODE="single_run2"
    elif [[ " ${ERAs_RUN3[*]} " =~ " ${SINGLE_ERA} " ]]; then
        EFFECTIVE_MODE="single_run3"
    else
        echo "ERROR: Unknown era '$SINGLE_ERA'"
        echo "Valid eras: ${ERAs_RUN2[*]} ${ERAs_RUN3[*]}"
        exit 1
    fi
    echo "Single era mode: $SINGLE_ERA"
fi

submit_condor_dags "$EFFECTIVE_MODE" MASSPOINTs "$METHOD" "$BINNING" "$EXTRA_ARGS" "$SINGLE_ERA"

echo ""
echo "============================================================"
echo "HTCondor DAG submission complete!"
echo ""
echo "Each masspoint has its own DAG with the full workflow:"
echo "  templates -> validate -> datacard -> combine -> asymptotic"
if [[ "$START_FROM" != "template" ]]; then
    echo "  (steps before '$START_FROM' marked as DONE)"
fi
echo ""
echo "Monitor progress with:"
echo "  condor_q -dag                    # Overall status"
echo "  condor_watch_q                   # Live monitor"
echo "  condor_q -dag -nobatch           # Detailed view"
echo "============================================================"
