#!/usr/bin/env python
"""
saveDataset.py - Create datasets for Mass-Decorrelated ParticleNet

Creates PyTorch Geometric datasets with:
- OS muon pair masses (mass1, mass2) for decorrelation
- Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]

Supports 4 sample types:
  signal    - Tight+Bjet, standard conversion
  ttX       - Tight+Bjet, standard conversion
  nonprompt - LNT+Bjet, ptCorr leptons, FR weights from correctionlib
  diboson   - 0-tag promoted, calibration weights from conditional_tables.json
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import argparse
import logging
import json
from datetime import datetime

import torch
import ROOT

from Preprocess import (GraphDataset, rtfileToDataList,
                        rtfileToDataList_nonprompt, rtfileToDataList_diboson,
                        load_fakerate_tables, load_conditional_tables)

parser = argparse.ArgumentParser()
parser.add_argument("--sample", required=True, type=str, help="MC sample name")
parser.add_argument("--sample-type", required=True,
                    choices=["signal", "nonprompt", "diboson", "ttX"],
                    help="sample type (determines conversion function)")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--eras", default="all", choices=["all", "run2", "run3"],
                    help="which eras to process (default: all)")
parser.add_argument("--output-name", default=None, type=str,
                    help="override output directory name (for era-split merging)")
parser.add_argument("--append", action="store_true", default=False,
                    help="append to existing fold files instead of overwriting")
parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
args = parser.parse_args()

# Check arguments
valid_channels = ["Run1E2Mu", "Run3Mu"]
if args.channel not in valid_channels:
    raise ValueError(f"Invalid channel {args.channel}. Valid options: {valid_channels}\n"
                     "Note: 'Combined' is handled by DynamicDatasetLoader at training time.")

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.environ["WORKDIR"]

# ---------------------------------------------------------------------------
# Era selection
# ---------------------------------------------------------------------------
RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]
ALL_ERAS = RUN2_ERAS + RUN3_ERAS

if args.eras == "run2":
    selected_eras = RUN2_ERAS
elif args.eras == "run3":
    selected_eras = RUN3_ERAS
else:
    selected_eras = ALL_ERAS

# ---------------------------------------------------------------------------
# Load external data for augmented sample types
# ---------------------------------------------------------------------------
fr_tables = None
cond_tables = None

if args.sample_type == "nonprompt":
    # Load correctionlib fake rate tables
    sknano_data = os.environ.get("SKNANO_DATA", "")
    if not sknano_data:
        sknano_data = os.path.join(
            os.path.dirname(WORKDIR), "SKNanoAnalyzer", "data",
            "Run3_v13_Run2_v9")
    if not os.path.isdir(sknano_data):
        raise RuntimeError(
            f"Cannot find correctionlib data at {sknano_data}. "
            "Set SKNANO_DATA environment variable.")
    logging.info(f"Loading fake rate tables from {sknano_data}")
    fr_tables = load_fakerate_tables(sknano_data)
    logging.info(f"Loaded fake rates for {len(fr_tables)} eras")

elif args.sample_type == "diboson":
    # Load conditional promotion tables
    cond_tables_path = os.path.join(
        WORKDIR, "ParticleNetMD", "DataAugment", "diboson",
        "plots", "rank_promote", "conditional_tables.json")
    if not os.path.exists(cond_tables_path):
        raise FileNotFoundError(
            f"conditional_tables.json not found at {cond_tables_path}. "
            "Run dibosonRankPromote.py first.")
    logging.info(f"Loading conditional tables from {cond_tables_path}")
    cond_tables = load_conditional_tables(cond_tables_path)
    logging.info("Loaded conditional tables for Run2 and Run3")

# ---------------------------------------------------------------------------
# Initialize statistics
# ---------------------------------------------------------------------------
dataset_stats = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "sample": args.sample,
    "sample_type": args.sample_type,
    "channel": args.channel,
    "eras_processed": args.eras,
    "node_features": "[E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]",
    "mass_decorrelation": True,
    "eras": {},
    "folds": {},
    "total_events": 0,
    "total_weight": 0.0
}

nFolds = 5
dataList = [[] for _ in range(nFolds)]

# ---------------------------------------------------------------------------
# Process data for each era
# ---------------------------------------------------------------------------
entry_offset = 0  # For diboson deterministic seeding across files

for era in selected_eras:
    is_run3 = era in RUN3_ERAS
    era_group = "Run3" if is_run3 else "Run2"

    file_path = f"{WORKDIR}/SKNanoOutput/EvtTreeProducer/{args.channel}/{era}/{args.sample}.root"

    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        continue

    logging.info(f"Processing {file_path}")
    rt = ROOT.TFile.Open(file_path)

    if args.sample_type in ("signal", "ttX"):
        sampleDataTmp = rtfileToDataList(rt, args.sample, args.channel, era,
                                         maxSize=-1, nFolds=nFolds)
    elif args.sample_type == "nonprompt":
        sampleDataTmp = rtfileToDataList_nonprompt(
            rt, args.sample, args.channel, era,
            fr_tables, is_run3, maxSize=-1, nFolds=nFolds)
    elif args.sample_type == "diboson":
        # Count entries for offset tracking
        n_entries = rt.Events.GetEntries()
        sampleDataTmp = rtfileToDataList_diboson(
            rt, args.sample, args.channel, era,
            cond_tables, era_group, entry_offset=entry_offset,
            maxSize=-1, nFolds=nFolds)
        entry_offset += n_entries

    rt.Close()

    # Merge into main data list
    for i in range(nFolds):
        dataList[i] += sampleDataTmp[i]

    # Calculate statistics
    total_events = sum(len(fold) for fold in sampleDataTmp)
    total_weight = sum(sum(data.weight.item() for data in fold) for fold in sampleDataTmp)

    dataset_stats["eras"][era] = {
        "events": total_events,
        "total_weight": total_weight
    }

    logging.info(f"Loaded {total_events} events (total weight: {total_weight:.2f}) from {era}/{args.channel}")

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
output_name = args.output_name if args.output_name else args.sample

if args.sample_type == "signal":
    baseDir = f"{WORKDIR}/ParticleNetMD/dataset/samples/signals/{output_name}"
else:
    baseDir = f"{WORKDIR}/ParticleNetMD/dataset/samples/backgrounds/{output_name}"

os.makedirs(baseDir, exist_ok=True)

# ---------------------------------------------------------------------------
# Save each fold (with optional append mode)
# ---------------------------------------------------------------------------
for i, data in enumerate(dataList):
    if len(data) == 0:
        continue

    filename = f"{baseDir}/{args.channel}_fold-{i}.pt"

    if args.append and os.path.exists(filename):
        # Load existing fold and extend
        existing_dataset = torch.load(filename, weights_only=False)
        existing_data = existing_dataset.data_list if hasattr(existing_dataset, 'data_list') else []
        combined_data = existing_data + data
        logging.info(f"Appending {len(data)} events to existing {len(existing_data)} in {filename}")
        graphdataset = GraphDataset(combined_data)
    else:
        graphdataset = GraphDataset(data)

    torch.save(graphdataset, filename)
    logging.info(f"Saved {len(graphdataset.data_list)} events to {filename}")

# ---------------------------------------------------------------------------
# Fold statistics
# ---------------------------------------------------------------------------
for i in range(nFolds):
    fold_events = len(dataList[i])
    fold_weight = sum(data.weight.item() for data in dataList[i]) if fold_events > 0 else 0.0
    dataset_stats["folds"][f"fold_{i}"] = {
        "events": fold_events,
        "total_weight": fold_weight
    }

dataset_stats["total_events"] = sum(len(fold) for fold in dataList)
dataset_stats["total_weight"] = sum(sum(data.weight.item() for data in fold) for fold in dataList)

logging.info("Finished loading dataset")

# Save dataset statistics
stats_filename = f"{baseDir}/{args.channel}_stats.json"
with open(stats_filename, 'w') as f:
    json.dump(dataset_stats, f, indent=2)
logging.info(f"Saved dataset statistics to {stats_filename}")

# Save summary log
log_dir = f"{WORKDIR}/ParticleNetMD/logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/dataset_stats_{args.sample_type}_{output_name}_{args.channel}_{timestamp}.json"
with open(log_filename, 'w') as f:
    json.dump(dataset_stats, f, indent=2)
logging.info(f"Saved dataset statistics log to {log_filename}")

logging.info("Finished saving dataset")
