#!/usr/bin/env python3
import os
import argparse
import json
import ROOT
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True, help="2016preVFP, 2016postVFP, 2017, 2018, FullRun2")
parser.add_argument("--channel", type=str, required=True, help="SR1E2Mu, SR3Mu, Combined")
parser.add_argument("--method", type=str, required=True, help="Baseline, ParticleNet")
parser.add_argument("--debug", action='store_true', default=False, help="Enable debug logging")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

if args.method == "Baseline":
    MASSPOINTs = [
        "MHc70_MA18", "MHc70_MA40", "MHc70_MA55", "MHc70_MA65",
        "MHc85_MA70", "MHc85_MA80",      # 85_21 is missng in 16a
        "MHc100_MA15", "MHc100_MA24", "MHc100_MA60", "MHc100_MA75", "MHc100_MA95",
        "MHc115_MA27", "MHc115_MA87", "MHc115_MA110",
        "MHc130_MA30", "MHc130_MA83", "MHc130_MA90", "MHc130_MA100", "MHc130_MA125",
        "MHc145_MA35", "MHc145_MA92", "MHc145_MA140",
        "MHc160_MA50", "MHc160_MA85", "MHc160_MA98", "MHc160_MA120", "MHc160_MA135", "MHc160_MA155"
    ]
elif args.method == "ParticleNet":
    MASSPOINTs = [
        "MHc100_MA95", "MHc130_MA90", "MHc160_MA85", "MHc115_MA87", "MHc145_MA92", "MHc160_MA98"
    ]
    raise NotImplementedError("ParticleNet not yet implemented")
else:
    raise ValueError("Invalid method")
REFERENCE_XSEC = 5.0  # pb

def parseAsymptoticLimit(masspoint, method):
    base_dir = f"templates/{args.era}/{args.channel}/{masspoint}/Shape/{method}"
    f = ROOT.TFile.Open(f"{base_dir}/higgsCombineTest.AsymptoticLimits.mH120.root")
    limit = f.Get("limit")
    xsecs = {}
    for idx, entry in enumerate(limit):
        xsecs[idx] = entry.limit * REFERENCE_XSEC
    f.Close()
    
    out = {}
    out["exp-2"] = xsecs[0]
    out["exp-1"] = xsecs[1]
    out["exp0"] = xsecs[2]
    out["exp+1"] = xsecs[3]
    out["exp+2"] = xsecs[4]
    out["obs"] = xsecs[5]

    return out

if __name__ == "__main__":
    limits = {}
    for masspoint in MASSPOINTs:
        limits[masspoint] = parseAsymptoticLimit(masspoint, args.method)
    
    outpath = f"results/json/limits.{args.era}.{args.channel}.Asymptotic.{args.method}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, "w") as f:
        json.dump(limits, f, indent=4)
        
