#!/usr/bin/env python
"""
saveDataset.py - Create datasets for Mass-Decorrelated ParticleNet

Creates PyTorch Geometric datasets with:
- OS muon pair masses (mass1, mass2) for decorrelation
- Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
"""
import os
import argparse
import logging
import json
from datetime import datetime

import torch
import ROOT

from Preprocess import GraphDataset, rtfileToDataList

parser = argparse.ArgumentParser()
parser.add_argument("--sample", required=True, type=str, help="MC sample name")
parser.add_argument("--sample-type", required=True, choices=["signal", "background"], help="sample type")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--pilot", action="store_true", default=False, help="pilot mode")
parser.add_argument("--debug", action="store_true", default=False, help="debug mode")
args = parser.parse_args()

# Check arguments
valid_channels = ["Run1E2Mu", "Run3Mu", "Combined"]

if args.channel not in valid_channels:
    raise ValueError(f"Invalid channel {args.channel}. Valid options: {valid_channels}")

# No validation of sample names - use MC sample names directly

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.environ["WORKDIR"]

# Initialize dataset statistics dictionary
dataset_stats = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "sample": args.sample,
    "sample_type": args.sample_type,
    "channel": args.channel,
    "pilot_mode": args.pilot,
    "node_features": "[E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]",
    "mass_decorrelation": True,
    "eras": {},
    "folds": {},
    "total_events": 0,
    "total_weight": 0.0
}

# Load dataset parameters
maxSize = 10000 if args.pilot else -1
nFolds = 5
maxSizeForEra = {
    "2016preVFP": int(maxSize / 7) if maxSize > 0 else -1,
    "2016postVFP": int(maxSize / 7) if maxSize > 0 else -1,
    "2017": int(maxSize * (2/7)) if maxSize > 0 else -1,
    "2018": int(maxSize * (3/7)) if maxSize > 0 else -1
}
logging.debug(maxSizeForEra)

# Initialize data lists
dataList = [[] for _ in range(nFolds)]

# Process channels to process
channels_to_process = ["Run1E2Mu", "Run3Mu"] if args.channel == "Combined" else [args.channel]

# Process data for each era
for era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    # Initialize era statistics
    if era not in dataset_stats["eras"]:
        dataset_stats["eras"][era] = {"events": 0, "total_weight": 0.0}

    for channel in channels_to_process:
        # Use sample name directly as filename (no mapping needed)
        file_path = f"{WORKDIR}/SKNanoOutput/EvtTreeProducer/{channel}/{era}/{args.sample}.root"

        if os.path.exists(file_path):
            logging.info(f"Processing {file_path}")
            rt = ROOT.TFile.Open(file_path)

            # Convert events to graph data (with masses for decorrelation)
            sampleDataTmp = rtfileToDataList(rt, args.sample, channel, era,
                                           maxSize=maxSizeForEra[era], nFolds=nFolds)
            rt.Close()

            # Merge into main data list
            for i in range(nFolds):
                dataList[i] += sampleDataTmp[i]

            # Calculate statistics
            total_events = sum(len(fold) for fold in sampleDataTmp)
            total_weight = sum(sum(data.weight.item() for data in fold) for fold in sampleDataTmp)

            dataset_stats["eras"][era]["events"] += total_events
            dataset_stats["eras"][era]["total_weight"] += total_weight

            logging.info(f"Loaded {total_events} events (total weight: {total_weight:.2f}) from {era}/{channel}")
        else:
            logging.warning(f"File not found: {file_path}")

# Calculate fold statistics
for i in range(nFolds):
    fold_events = len(dataList[i])
    fold_weight = sum(data.weight.item() for data in dataList[i]) if fold_events > 0 else 0.0
    dataset_stats["folds"][f"fold_{i}"] = {
        "events": fold_events,
        "total_weight": fold_weight
    }

# Calculate total statistics
dataset_stats["total_events"] = sum(len(fold) for fold in dataList)
dataset_stats["total_weight"] = sum(sum(data.weight.item() for data in fold) for fold in dataList)

logging.info("Finished loading dataset")

# Save datasets with MC sample name structure
if args.sample_type == "signal":
    baseDir = f"{WORKDIR}/ParticleNetMD/dataset/samples/signals/{args.sample}"
else:
    baseDir = f"{WORKDIR}/ParticleNetMD/dataset/samples/backgrounds/{args.sample}"

if args.pilot:
    baseDir += "_pilot"

logging.info(f"Saving dataset to {baseDir}")
os.makedirs(baseDir, exist_ok=True)

# Save each fold
for i, data in enumerate(dataList):
    if len(data) > 0:  # Only save non-empty folds
        graphdataset = GraphDataset(data)
        filename = f"{baseDir}/{args.channel}_fold-{i}.pt"
        torch.save(graphdataset, filename)
        logging.info(f"Saved {len(data)} events to {filename}")

# Save dataset statistics to JSON file
stats_filename = f"{baseDir}/{args.channel}_stats.json"
with open(stats_filename, 'w') as f:
    json.dump(dataset_stats, f, indent=2)
logging.info(f"Saved dataset statistics to {stats_filename}")

# Also save a summary log in the main logs directory
log_dir = f"{WORKDIR}/ParticleNetMD/logs"
os.makedirs(log_dir, exist_ok=True)

# Create a unique log filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/dataset_stats_{args.sample_type}_{args.sample}_{args.channel}_{timestamp}.json"

with open(log_filename, 'w') as f:
    json.dump(dataset_stats, f, indent=2)
logging.info(f"Saved dataset statistics log to {log_filename}")

logging.info("Finished saving dataset")