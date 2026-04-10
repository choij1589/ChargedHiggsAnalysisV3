#!/bin/bash
# Parse per-era nonprompt normalization uncertainties from closure test results
# and write them to Common/Data/FakeNorm.json.
#
# Source of uncertainty:
#   Run1E2Mu : pair/mass   Central  -> recommended_systematic_pct
#   Run3Mu   : max(pair_lowM/mass, pair_highM/mass)  Central

set -euo pipefail
python python/parseFakeNorm.py
