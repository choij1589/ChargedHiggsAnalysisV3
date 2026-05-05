#!/bin/bash
#
# test_dropRun3_compare.sh
#
# Print side-by-side asymptotic-limit comparison between the full
# 'All' combination and the 'All_drop_16post_22_23BPix' test, for the
# 4 test mass points.
#
# Usage:
#   bash scripts/test_dropRun3_compare.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(dirname "$SCRIPT_DIR")"

TEST_ERA="All_drop_16post_22_23BPix"
REF_ERA="All"
CHANNEL="Combined"
METHOD="Baseline"
BINNING="extended"

MASSPOINTS=(
    "MHc70_MA15"
    "MHc130_MA55"
    "MHc100_MA95"
    "MHc160_MA155"
)

cd "$MODULE_DIR"

limit_file() {
    local era="$1" mp="$2"
    echo "templates/${era}/${CHANNEL}/${mp}/${METHOD}/${BINNING}/combine_output/asymptotic/higgsCombine.${mp}.${METHOD}.${BINNING}.AsymptoticLimits.mH120.root"
}

printf "%-16s  %-14s  %10s  %10s  %10s  %10s  %10s  %10s\n" \
    "MassPoint" "Era" "Exp-2σ" "Exp-1σ" "Exp med" "Exp+1σ" "Exp+2σ" "Obs"

for MP in "${MASSPOINTS[@]}"; do
    for ERA in "$REF_ERA" "$TEST_ERA"; do
        F="$(limit_file "$ERA" "$MP")"
        if [[ ! -f "$F" ]]; then
            printf "%-16s  %-14s  %s\n" "$MP" "$ERA" "(missing: $F)"
            continue
        fi

        VALS=$(root -l -b -q -e "
            TFile *f = TFile::Open(\"${F}\");
            TTree *t = (TTree*)f->Get(\"limit\");
            double r;
            t->SetBranchAddress(\"limit\", &r);
            for (int i = 0; i < t->GetEntries(); i++) { t->GetEntry(i); printf(\"%.5f \", r); }
            f->Close();
        " 2>/dev/null | tr -d '\n' | awk '{print $1, $2, $3, $4, $5, $6}')

        # shellcheck disable=SC2086
        set -- $VALS
        printf "%-16s  %-14s  %10s  %10s  %10s  %10s  %10s  %10s\n" \
            "$MP" "$ERA" "${1:-n/a}" "${2:-n/a}" "${3:-n/a}" "${4:-n/a}" "${5:-n/a}" "${6:-n/a}"
    done
    echo "----------------------------------------------------------------------------------------------"
done
