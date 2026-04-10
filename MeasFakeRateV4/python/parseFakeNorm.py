#!/usr/bin/env python3
"""
Parse per-era nonprompt normalization uncertainties from closure test results
and write them to Common/Data/FakeNorm.json.

Source of uncertainty (Central syst only):
  Run1E2Mu : pair/mass
  Run3Mu   : max(pair_lowM/mass, pair_highM/mass)

Output structure:
  {
    "description": "...",
    "units": "fraction (e.g. 0.20 = 20%)",
    "Run1E2Mu": { "2016preVFP": 0.10, "2016postVFP": 0.10, ... },
    "Run3Mu":   { "2016preVFP": 0.25, "2022EE": 0.35, ... }
  }
"""
import os
import json

WORKDIR = os.environ["WORKDIR"]
PLOTS   = f"{WORKDIR}/MeasFakeRateV4/plots"
OUTPUT  = f"{WORKDIR}/Common/Data/FakeNorm.json"

ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022", "2022EE", "2023", "2023BPix",
]


def load_syst_pct(era, channel, histkey):
    variable = histkey.replace("/", "_").lower()
    path = f"{PLOTS}/{era}/{channel}/Central/closure_{variable}_yield.json"
    with open(path) as f:
        data = json.load(f)
    if "recommended_systematic_pct" not in data:
        raise KeyError(f"No recommended_systematic_pct in {path} — re-run plotClosure.py")
    return data["recommended_systematic_pct"]


result = {
    "description": (
        "Per-era nonprompt normalization uncertainties derived from closure tests. "
        "Run1E2Mu uses pair/mass; Run3Mu uses max(pair_lowM/mass, pair_highM/mass). "
        "Central syst variation only."
    ),
    "units": "fraction (e.g. 0.20 = 20%)",
    "Run1E2Mu": {},
    "Run3Mu": {},
}

for era in ERAS:
    # --- Run1E2Mu: pair/mass ---
    syst_pct = load_syst_pct(era, "Run1E2Mu", "pair/mass")
    result["Run1E2Mu"][era] = round(syst_pct / 100, 2)

    # --- Run3Mu: max(pair_lowM/mass, pair_highM/mass) ---
    low_pct  = load_syst_pct(era, "Run3Mu", "pair_lowM/mass")
    high_pct = load_syst_pct(era, "Run3Mu", "pair_highM/mass")
    result["Run3Mu"][era] = round(max(low_pct, high_pct) / 100, 2)

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, "w") as f:
    json.dump(result, f, indent=4)

print(f"Written: {OUTPUT}")
print()
print(f"{'Era':<15}  {'Run1E2Mu':>10}  {'Run3Mu':>10}")
print("-" * 40)
for era in ERAS:
    r1 = result["Run1E2Mu"][era]
    r3 = result["Run3Mu"][era]
    print(f"{era:<15}  {r1:>10.0%}  {r3:>10.0%}")
