#!/bin/bash
set -euo pipefail

MASSPOINTs=("MHc100_MA95" "MHc130_MA90" "MHc160_MA85")

SRCDIR="$(cd "$(dirname "$0")/.." && pwd)/GAOptim/Combined"
DSTDIR="$(cd "$(dirname "$0")/../.." && pwd)/SKNanoAnalyzer/data/Run3_v13_Run2_v9/All/Combined/Classifiers/ParticleNetMD"

for mp in "${MASSPOINTs[@]}"; do
    src="${SRCDIR}/${mp}/fold-4/best_model"
    dst="${DSTDIR}/${mp}/best_model"

    if [[ ! -d "$src" ]]; then
        echo "ERROR: source not found: $src"
        exit 1
    fi

    mkdir -p "$dst"
    cp -v "$src"/model.pt "$src"/model_info.json "$dst"/
    echo "Copied $mp"
done

echo "Done."
