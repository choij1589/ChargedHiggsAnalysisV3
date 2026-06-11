#!/bin/bash
set -euo pipefail

# Copy datacards and shape files to the datacards repo, organized by mass point.
# For ParticleNet-trained mass points, use ParticleNet method; otherwise use Baseline.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"
DATACARDS_REPO="${MODULE_DIR}/../../datacards"
TEMPLATE_BASE="${MODULE_DIR}/templates/All/Combined"

if [[ ! -d "$DATACARDS_REPO" ]]; then
    echo "ERROR: datacards repo not found at $DATACARDS_REPO"
    exit 1
fi

DEST_BASE="${DATACARDS_REPO}/input"

# Read mass point lists from configs/masspoints.json
BASELINE_MASSPOINTS=($(python3 -c "
import json
with open('${MODULE_DIR}/configs/masspoints.json') as f:
    data = json.load(f)
for mp in data['baseline']:
    print(mp)
"))

PARTICLENET_MASSPOINTS=($(python3 -c "
import json
with open('${MODULE_DIR}/configs/masspoints.json') as f:
    data = json.load(f)
for mp in data['particlenet']:
    print(mp)
"))

echo "Baseline mass points: ${#BASELINE_MASSPOINTS[@]}"
echo "ParticleNet mass points: ${#PARTICLENET_MASSPOINTS[@]}"

# Helper: check if a mass point is in the ParticleNet list
is_particlenet() {
    local mp="$1"
    for pn_mp in "${PARTICLENET_MASSPOINTS[@]}"; do
        if [[ "$mp" == "$pn_mp" ]]; then
            return 0
        fi
    done
    return 1
}

copied=0
errors=0

for mp in "${BASELINE_MASSPOINTS[@]}"; do
    if is_particlenet "$mp"; then
        method="ParticleNet"
    else
        method="Baseline"
    fi

    src_dir="${TEMPLATE_BASE}/${mp}/${method}/extended"
    dest_dir="${DEST_BASE}/${mp}"

    if [[ ! -d "$src_dir" ]]; then
        echo "WARNING: source dir not found: $src_dir"
        errors=$((errors + 1))
        continue
    fi

    mkdir -p "$dest_dir"

    # Copy datacard and shape files
    cp "$src_dir/datacard.txt" "$dest_dir/"
    cp "$src_dir"/shapes_*.root "$dest_dir/"

    n_files=$(ls "$dest_dir"/ | wc -l)
    echo "  $mp ($method): copied $n_files files"
    copied=$((copied + 1))
done

echo ""
echo "Done: $copied mass points copied, $errors errors"

if [[ $errors -gt 0 ]]; then
    exit 1
fi
