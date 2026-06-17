#!/bin/bash
#
# runImpactBreakdown.sh - Grouped nuisance uncertainty breakdown with Combine.
#
# The script creates a temporary datacard with nuisance group definitions and
# runs cumulative MultiDimFit scans using --freezeNuisanceGroups. Production
# datacards are not modified.
#
# Usage:
#   ./scripts/runImpactBreakdown.sh --era All --channel Combined \
#     --masspoint MHc130_MA90 --method Baseline --binning extended --unblind
#

set -euo pipefail

ERA=""
CHANNEL="Combined"
MASSPOINT=""
METHOD="Baseline"
BINNING="extended"
NUISANCE="fallback_lnn"
MASS="120"
POINTS="100"
RMIN="-5"
RMAX="5"
EXPECT_SIGNAL="1"
PARTIAL_UNBLIND=false
UNBLIND=false
DRY_RUN=false
VERBOSE=false
CONDOR=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --era) ERA="$2"; shift 2 ;;
        --channel) CHANNEL="$2"; shift 2 ;;
        --masspoint) MASSPOINT="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --binning) BINNING="$2"; shift 2 ;;
        --nuisance) NUISANCE="$2"; shift 2 ;;
        --mass) MASS="$2"; shift 2 ;;
        --points) POINTS="$2"; shift 2 ;;
        --rMin) RMIN="$2"; shift 2 ;;
        --rMax) RMAX="$2"; shift 2 ;;
        --r-range)
            IFS=',' read -r RMIN RMAX <<< "$2"
            shift 2
            ;;
        --expect-signal) EXPECT_SIGNAL="$2"; shift 2 ;;
        --partial-unblind) PARTIAL_UNBLIND=true; shift ;;
        --unblind) UNBLIND=true; shift ;;
        --condor) CONDOR=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --era ERA              Data-taking period"
            echo "  --channel CHANNEL      Analysis channel [default: Combined]"
            echo "  --masspoint MASSPOINT  Signal mass point, e.g. MHc130_MA90"
            echo ""
            echo "Options:"
            echo "  --method METHOD        Baseline or ParticleNet [default: Baseline]"
            echo "  --binning BINNING      Binning scheme [default: extended]"
            echo "  --nuisance MODE        fallback_lnn (default) or preserve_shape"
            echo "  --mass MASS            Combine mass label [default: 120]"
            echo "  --points N             Grid points per scan [default: 100]"
            echo "  --rMin VALUE           POI lower bound [default: -5]"
            echo "  --rMax VALUE           POI upper bound [default: 5]"
            echo "  --r-range MIN,MAX      POI range shortcut"
            echo "  --expect-signal VALUE  Asimov signal strength when blinded [default: 1]"
            echo "  --partial-unblind      Use partial-unblind template suffix and real data_obs"
            echo "  --unblind              Use full-unblind template suffix and real data_obs"
            echo "  --condor               Submit the grouped scans as an HTCondor DAG"
            echo "  --dry-run              Print commands without executing Combine"
            echo "  --verbose              Enable verbose logging"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era, --channel, and --masspoint are required"
    exit 1
fi
if [[ "$UNBLIND" == true && "$PARTIAL_UNBLIND" == true ]]; then
    echo "ERROR: --unblind and --partial-unblind are mutually exclusive"
    exit 1
fi
case "$NUISANCE" in
    fallback_lnn|preserve_shape) ;;
    *) echo "ERROR: Invalid --nuisance value '$NUISANCE'"; exit 1 ;;
esac
if [[ -z "$RMIN" || -z "$RMAX" ]]; then
    echo "ERROR: --r-range must be formatted as MIN,MAX"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

BINNING_SUFFIX="${BINNING}"
if [[ "$UNBLIND" == true ]]; then
    BINNING_SUFFIX="${BINNING}_unblind"
elif [[ "$PARTIAL_UNBLIND" == true ]]; then
    BINNING_SUFFIX="${BINNING}_partial_unblind"
fi
if [[ "$NUISANCE" == "preserve_shape" ]]; then
    BINNING_SUFFIX="${BINNING_SUFFIX}_preserve_shape"
fi

TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV3/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING_SUFFIX}"
DATACARD="${TEMPLATE_DIR}/datacard.txt"
SHAPES="${TEMPLATE_DIR}/shapes.root"

if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "ERROR: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi
if [[ ! -f "$DATACARD" ]]; then
    echo "ERROR: Datacard not found: $DATACARD"
    exit 1
fi
if [[ ! -f "$SHAPES" ]]; then
    echo "ERROR: shapes.root not found: $SHAPES"
    exit 1
fi

if [[ "$UNBLIND" == true || "$PARTIAL_UNBLIND" == true ]]; then
    DATA_TAG="obs"
    ASIMOV_OPTIONS=""
else
    EXPECT_TAG="${EXPECT_SIGNAL//./p}"
    EXPECT_TAG="${EXPECT_TAG//-/m}"
    DATA_TAG="r${EXPECT_TAG}"
    ASIMOV_OPTIONS="-t -1 --expectSignal ${EXPECT_SIGNAL}"
fi

OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/impact_breakdown_${DATA_TAG}"
GROUPED_DATACARD="${OUTPUT_DIR}/grouped_datacard.txt"
GROUPED_WORKSPACE="${OUTPUT_DIR}/grouped_workspace.root"
GROUP_JSON="${OUTPUT_DIR}/group_members.json"
FREEZE_GROUPS_FILE="${OUTPUT_DIR}/freeze_groups.txt"
BREAKDOWN_LABELS_FILE="${OUTPUT_DIR}/breakdown_labels.txt"
TAG="${MASSPOINT}.${METHOD}.${BINNING_SUFFIX}.ImpactBreakdown"
TOTAL_NAME=".${TAG}.total"
TOTAL_FILE="higgsCombine.${TAG}.total.MultiDimFit.mH${MASS}.root"

log() { [[ "$VERBOSE" == true ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" || true; }
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $1"
    else
        log "Running: $1"
        eval "$1"
    fi
}

echo "Running impact breakdown for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})"
echo "  Output: ${OUTPUT_DIR}"
echo "  r range: ${RMIN},${RMAX}"
echo "  Grid points: ${POINTS}"
echo "  Condor: ${CONDOR}"
echo "  Data mode: $(if [[ -n "$ASIMOV_OPTIONS" ]]; then echo "Asimov (expectSignal=${EXPECT_SIGNAL})"; else echo 'Observed (real data)'; fi)"

mkdir -p "$OUTPUT_DIR"

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY-RUN] Would create grouped datacard: ${GROUPED_DATACARD}"
    echo "[DRY-RUN] Would link shapes.root into: ${OUTPUT_DIR}"
else
    ln -sf "$SHAPES" "${OUTPUT_DIR}/shapes.root"
fi

python3 - "$DATACARD" "$GROUPED_DATACARD" "$GROUP_JSON" "$FREEZE_GROUPS_FILE" "$BREAKDOWN_LABELS_FILE" "$DRY_RUN" <<'PYEOF'
import json
import os
import re
import sys
from collections import OrderedDict

datacard, grouped_datacard, group_json, freeze_groups_file, labels_file, dry_run = sys.argv[1:7]
dry_run = dry_run == "true"

group_specs = OrderedDict([
    ("signal_theory", [
        r"^QCDScale_mu[RF]_BSMsignal_",
        r"^pdf(_alphas)?_",
        r"^ps_(isr|fsr)_",
    ]),
    ("prompt_norm", [
        r"^CMS_B2G25013_Norm_(WZ|ZZ|ttW|ttZ|ttH|tZq|conversion|others)_",
    ]),
    ("nonprompt_norm", [
        r"^CMS_B2G25013_Norm_nonprompt_",
    ]),
    ("experimental", [
        r"^CMS_scale_m_",
        r"^CMS_scale_j_",
        r"^CMS_res_j_",
        r"^CMS_scale_met_unclustered_energy_",
        r"^CMS_btag_",
        r"^lumi",
    ]),
])

compiled = [(group, [re.compile(pat) for pat in patterns]) for group, patterns in group_specs.items()]
skip_first_tokens = {
    "imax", "jmax", "kmax", "shapes", "bin", "observation", "process",
    "rate", "------------", "--------------------------------------------------------------------------------",
}
skip_types = {"rateParam", "flatParam", "extArg", "autoMCStats", "group"}
constrained_types = {"lnN", "lnU", "shape", "shape?", "gmN", "gmM", "param", "constr"}

with open(datacard) as handle:
    lines = handle.readlines()

nuisances = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        continue
    fields = stripped.split()
    if len(fields) < 2:
        continue
    if fields[0] in skip_first_tokens:
        continue
    nuisance_type = fields[1]
    if nuisance_type in skip_types:
        continue
    if nuisance_type not in constrained_types:
        continue
    nuisances.append(fields[0])

groups = OrderedDict((group, []) for group in group_specs)

for nuisance in nuisances:
    matched = False
    for group, patterns in compiled:
        if any(pattern.search(nuisance) for pattern in patterns):
            groups[group].append(nuisance)
            matched = True
            break
    if not matched:
        groups["experimental"].append(nuisance)

active_freeze_groups = [group for group in group_specs if groups[group]]
breakdown_labels = active_freeze_groups + ["stat"]

summary = OrderedDict()
for group, members in groups.items():
    summary[group] = {
        "count": len(members),
        "members": members,
    }

print("Nuisance group summary:")
for group, payload in summary.items():
    print(f"  {group}: {payload['count']}")

if dry_run:
    sys.exit(0)

os.makedirs(os.path.dirname(grouped_datacard), exist_ok=True)
with open(grouped_datacard, "w") as out:
    out.writelines(lines)
    if lines and not lines[-1].endswith("\n"):
        out.write("\n")
    out.write("\n# Nuisance groups for grouped impact breakdown\n")
    for group, payload in summary.items():
        members = payload["members"]
        if members:
            out.write(f"{group} group = {' '.join(members)}\n")

with open(group_json, "w") as out:
    json.dump(summary, out, indent=2)
with open(freeze_groups_file, "w") as out:
    for group in active_freeze_groups:
        out.write(group + "\n")
with open(labels_file, "w") as out:
    out.write(",".join(breakdown_labels) + "\n")
PYEOF

if [[ "$DRY_RUN" == true ]]; then
    FREEZE_GROUPS=(signal_theory prompt_norm nonprompt_norm experimental)
    BREAKDOWN_LABELS="signal_theory,prompt_norm,nonprompt_norm,experimental,stat"
else
    mapfile -t FREEZE_GROUPS < "$FREEZE_GROUPS_FILE"
    BREAKDOWN_LABELS="$(< "$BREAKDOWN_LABELS_FILE")"
fi

cd "$OUTPUT_DIR"

# ============================================================
# Condor DAG execution
# ============================================================
if [[ "$CONDOR" == true ]]; then
    CONDOR_DIR="${OUTPUT_DIR}/condor"
    CMSSW_BASE="${WORKDIR}/Common/CMSSW_14_1_0_pre4/src"
    mkdir -p "${CONDOR_DIR}/logs"

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] Would create workspace locally for Condor transfer:"
        echo "[DRY-RUN]   text2workspace.py grouped_datacard.txt -o grouped_workspace.root"
        echo "[DRY-RUN] Would create and submit DAG under: ${CONDOR_DIR}"
        exit 0
    fi

    echo "Creating grouped workspace locally for Condor transfer..."
    text2workspace.py grouped_datacard.txt -o grouped_workspace.root 2>&1 | tee text2workspace.out
    cp grouped_workspace.root "${CONDOR_DIR}/"

    cat > "${CONDOR_DIR}/total_scan.sh" << EOFTOTAL
#!/bin/bash
set -e
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
combine -M MultiDimFit grouped_workspace.root \\
    --algo grid \\
    --points ${POINTS} \\
    --setParameterRanges r=${RMIN},${RMAX} \\
    --saveWorkspace \\
    -n ${TOTAL_NAME} \\
    -m ${MASS} \\
    ${ASIMOV_OPTIONS} \\
    2>&1 | tee total_scan.out
EOFTOTAL
    chmod +x "${CONDOR_DIR}/total_scan.sh"

    cat > "${CONDOR_DIR}/total_scan.sub" << EOF
universe = vanilla
executable = ${CONDOR_DIR}/total_scan.sh
output = ${CONDOR_DIR}/logs/total_scan.out
error = ${CONDOR_DIR}/logs/total_scan.err
log = ${CONDOR_DIR}/breakdown.log

request_cpus = 1
request_memory = 4GB
request_disk = 1GB

should_transfer_files = YES
transfer_input_files = ${CONDOR_DIR}/grouped_workspace.root
transfer_output_files = ${TOTAL_FILE},total_scan.out
when_to_transfer_output = ON_EXIT

queue
EOF

    OTHER_ARGS=()
    COLORS=(2 4 6 8 9 28 46 38)
    CUMULATIVE_GROUPS=()
    FREEZE_JOB_NAMES=()
    FREEZE_OUTPUT_FILES="${CONDOR_DIR}/${TOTAL_FILE}"

    for IDX in "${!FREEZE_GROUPS[@]}"; do
        GROUP="${FREEZE_GROUPS[$IDX]}"
        CUMULATIVE_GROUPS+=("$GROUP")
        CUMULATIVE_CSV="$(IFS=','; echo "${CUMULATIVE_GROUPS[*]}")"
        CUMULATIVE_TAG="${CUMULATIVE_CSV//,/_}"
        FREEZE_NAME=".${TAG}.freeze_${CUMULATIVE_TAG}"
        FREEZE_FILE="higgsCombine.${TAG}.freeze_${CUMULATIVE_TAG}.MultiDimFit.mH${MASS}.root"
        JOB_NAME="freeze_${CUMULATIVE_TAG}"
        COLOR="${COLORS[$IDX]}"
        EXTRA_RTD=""
        if [[ "$IDX" -eq $((${#FREEZE_GROUPS[@]} - 1)) ]]; then
            EXTRA_RTD="--X-rtd MINIMIZER_no_analytic"
        fi
        FREEZE_JOB_NAMES+=("$JOB_NAME")
        FREEZE_OUTPUT_FILES="${FREEZE_OUTPUT_FILES},${CONDOR_DIR}/${FREEZE_FILE}"
        OTHER_ARGS+=("${FREEZE_FILE}:${GROUP}:${COLOR}")

        cat > "${CONDOR_DIR}/${JOB_NAME}.sh" << EOFFREEZE
#!/bin/bash
set -e
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
combine -M MultiDimFit ${TOTAL_FILE} \\
    --snapshotName MultiDimFit \\
    --algo grid \\
    --points ${POINTS} \\
    --setParameterRanges r=${RMIN},${RMAX} \\
    --freezeNuisanceGroups ${CUMULATIVE_CSV} \\
    ${EXTRA_RTD} \\
    -n ${FREEZE_NAME} \\
    -m ${MASS} \\
    ${ASIMOV_OPTIONS} \\
    2>&1 | tee ${JOB_NAME}.out
EOFFREEZE
        chmod +x "${CONDOR_DIR}/${JOB_NAME}.sh"

        cat > "${CONDOR_DIR}/${JOB_NAME}.sub" << EOF
universe = vanilla
executable = ${CONDOR_DIR}/${JOB_NAME}.sh
output = ${CONDOR_DIR}/logs/${JOB_NAME}.out
error = ${CONDOR_DIR}/logs/${JOB_NAME}.err
log = ${CONDOR_DIR}/breakdown.log

request_cpus = 1
request_memory = 4GB
request_disk = 1GB

should_transfer_files = YES
transfer_input_files = ${CONDOR_DIR}/${TOTAL_FILE}
transfer_output_files = ${FREEZE_FILE},${JOB_NAME}.out
when_to_transfer_output = ON_EXIT

queue
EOF
    done

    cat > "${CONDOR_DIR}/plot_breakdown.sh" << EOFPLOT
#!/bin/bash
set -e
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}
plot1DScan.py ${TOTAL_FILE} \\
    --main-label Total \\
    --main-color 1 \\
    --POI r \\
    --others ${OTHER_ARGS[*]} \\
    --breakdown ${BREAKDOWN_LABELS} \\
    --y-cut 100 \\
    --y-max 10 \\
    -o breakdown \\
    2>&1 | tee plot.out
EOFPLOT
    chmod +x "${CONDOR_DIR}/plot_breakdown.sh"

    cat > "${CONDOR_DIR}/plot_breakdown.sub" << EOF
universe = vanilla
executable = ${CONDOR_DIR}/plot_breakdown.sh
output = ${CONDOR_DIR}/logs/plot_breakdown.out
error = ${CONDOR_DIR}/logs/plot_breakdown.err
log = ${CONDOR_DIR}/breakdown.log

request_cpus = 1
request_memory = 2GB
request_disk = 1GB

should_transfer_files = YES
transfer_input_files = ${FREEZE_OUTPUT_FILES}
transfer_output_files = breakdown.pdf,breakdown.png,breakdown.root,plot.out
when_to_transfer_output = ON_EXIT

queue
EOF

    cat > "${CONDOR_DIR}/copy_outputs.sh" << EOFCOPY
#!/bin/bash
set -e
cp -f "${CONDOR_DIR}/breakdown.pdf" "${OUTPUT_DIR}/" 2>/dev/null || true
cp -f "${CONDOR_DIR}/breakdown.png" "${OUTPUT_DIR}/" 2>/dev/null || true
cp -f "${CONDOR_DIR}/breakdown.root" "${OUTPUT_DIR}/" 2>/dev/null || true
cp -f "${CONDOR_DIR}/plot.out" "${OUTPUT_DIR}/" 2>/dev/null || true
echo "Copied breakdown outputs to ${OUTPUT_DIR}"
EOFCOPY
    chmod +x "${CONDOR_DIR}/copy_outputs.sh"

    {
        echo "# Impact breakdown DAG"
        echo "# Generated for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})"
        echo "CONFIG ${SCRIPT_DIR}/../configs/dagman.config"
        echo ""
        echo "JOB total_scan total_scan.sub"
        for JOB_NAME in "${FREEZE_JOB_NAMES[@]}"; do
            echo "JOB ${JOB_NAME} ${JOB_NAME}.sub"
        done
        echo "JOB plot_breakdown plot_breakdown.sub"
        echo ""
        echo "PARENT total_scan CHILD ${FREEZE_JOB_NAMES[*]}"
        echo "PARENT ${FREEZE_JOB_NAMES[*]} CHILD plot_breakdown"
        echo "SCRIPT POST plot_breakdown copy_outputs.sh"
    } > "${CONDOR_DIR}/breakdown.dag"

    echo "Submitting impact breakdown DAG from ${CONDOR_DIR}"
    cd "$CONDOR_DIR"
    condor_submit_dag -f breakdown.dag
    echo "Monitor with: condor_q -dag"
    echo "DAG log: ${CONDOR_DIR}/breakdown.dag.dagman.out"
    exit 0
fi

# ============================================================
# Local execution
# ============================================================
run_cmd "text2workspace.py grouped_datacard.txt -o grouped_workspace.root 2>&1 | tee text2workspace.out"

TOTAL_CMD="combine -M MultiDimFit grouped_workspace.root \
    --algo grid \
    --points ${POINTS} \
    --setParameterRanges r=${RMIN},${RMAX} \
    --saveWorkspace \
    -n ${TOTAL_NAME} \
    -m ${MASS} \
    ${ASIMOV_OPTIONS} \
    2>&1 | tee total_scan.out"
run_cmd "$TOTAL_CMD"

OTHER_ARGS=()
COLOR_INDEX=0
COLORS=(2 4 6 8 9 28 46 38)
CUMULATIVE_GROUPS=()

for GROUP in "${FREEZE_GROUPS[@]}"; do
    CUMULATIVE_GROUPS+=("$GROUP")
    CUMULATIVE_CSV="$(IFS=','; echo "${CUMULATIVE_GROUPS[*]}")"
    CUMULATIVE_TAG="${CUMULATIVE_CSV//,/_}"
    FREEZE_NAME=".${TAG}.freeze_${CUMULATIVE_TAG}"
    FREEZE_FILE="higgsCombine.${TAG}.freeze_${CUMULATIVE_TAG}.MultiDimFit.mH${MASS}.root"
    COLOR="${COLORS[$COLOR_INDEX]}"
    EXTRA_RTD=""
    if [[ "$COLOR_INDEX" -eq $((${#FREEZE_GROUPS[@]} - 1)) ]]; then
        EXTRA_RTD="--X-rtd MINIMIZER_no_analytic"
    fi
    COLOR_INDEX=$((COLOR_INDEX + 1))

    FREEZE_CMD="combine -M MultiDimFit ${TOTAL_FILE} \
        --snapshotName MultiDimFit \
        --algo grid \
        --points ${POINTS} \
        --setParameterRanges r=${RMIN},${RMAX} \
        --freezeNuisanceGroups ${CUMULATIVE_CSV} \
        ${EXTRA_RTD} \
        -n ${FREEZE_NAME} \
        -m ${MASS} \
        ${ASIMOV_OPTIONS} \
        2>&1 | tee freeze_${CUMULATIVE_TAG}.out"
    run_cmd "$FREEZE_CMD"
    OTHER_ARGS+=("${FREEZE_FILE}:${GROUP}:${COLOR}")
done

PLOT_CMD="plot1DScan.py ${TOTAL_FILE} \
    --main-label Total \
    --main-color 1 \
    --POI r \
    --others ${OTHER_ARGS[*]} \
    --breakdown ${BREAKDOWN_LABELS} \
    --y-cut 100 \
    --y-max 10 \
    -o breakdown \
    2>&1 | tee plot.out"
run_cmd "$PLOT_CMD"

if [[ "$DRY_RUN" == false ]]; then
    echo ""
    echo "SUCCESS: impact breakdown outputs saved to ${OUTPUT_DIR}"
    ls -lh breakdown.pdf breakdown.png group_members.json grouped_datacard.txt 2>/dev/null || true
fi
