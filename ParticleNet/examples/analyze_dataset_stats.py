#!/usr/bin/env python3
"""
Script to analyze dataset statistics from JSON logs
"""
import json
import argparse
import os
import glob
from tabulate import tabulate

def load_stats(json_file):
    """Load statistics from JSON file"""
    with open(json_file, 'r') as f:
        return json.load(f)

def print_era_summary(stats):
    """Print summary of events per era"""
    print("\n=== Events per Era ===")
    era_data = []
    
    for era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
        if era in stats["eras"]:
            row = [era, stats["eras"][era]["signal"]]
            for bkg in ["nonprompt", "diboson", "ttZ"]:
                row.append(stats["eras"][era]["backgrounds"].get(bkg, 0))
            era_data.append(row)
    
    headers = ["Era", "Signal", "Nonprompt", "Diboson", "ttZ"]
    print(tabulate(era_data, headers=headers, tablefmt="grid"))

def print_fold_summary(stats):
    """Print summary of events per fold"""
    print("\n=== Events per Fold ===")
    
    # Before balancing
    print("\nBefore Balancing:")
    fold_data_before = []
    for i in range(5):
        fold_key = f"fold_{i}"
        if fold_key in stats["folds"]:
            fold_stats = stats["folds"][fold_key]["before_balancing"]
            row = [f"Fold {i}"]
            if "signal" in fold_stats:  # Binary classification
                row.extend([fold_stats.get("signal", 0), fold_stats.get(list(fold_stats.keys())[1], 0)])
            else:  # Multi-class
                row.extend([fold_stats.get(cls, 0) for cls in ["signal", "nonprompt", "diboson", "ttZ"]])
            fold_data_before.append(row)
    
    if stats["multiclass"]:
        headers = ["Fold", "Signal", "Nonprompt", "Diboson", "ttZ"]
    else:
        headers = ["Fold", "Signal", stats["background"]]
    print(tabulate(fold_data_before, headers=headers, tablefmt="grid"))
    
    # After balancing
    print("\nAfter Balancing:")
    fold_data_after = []
    for i in range(5):
        fold_key = f"fold_{i}"
        if fold_key in stats["folds"] and stats["folds"][fold_key]["after_balancing"]:
            min_size = stats["folds"][fold_key]["min_size"]
            row = [f"Fold {i}", min_size]
            fold_data_after.append(row)
    
    headers = ["Fold", "Events per class"]
    print(tabulate(fold_data_after, headers=headers, tablefmt="grid"))

def print_total_summary(stats):
    """Print total summary"""
    print("\n=== Total Events Summary ===")
    total = stats["total_events"]
    
    print(f"Total Signal Events: {total['signal']:,}")
    print(f"Total Background Events:")
    for bkg, count in total["backgrounds"].items():
        print(f"  - {bkg}: {count:,}")
    print(f"Grand Total: {total['total']:,}")
    
    # Calculate efficiency
    total_after = sum(stats["folds"][f"fold_{i}"]["min_size"] * len(stats["folds"][f"fold_{i}"]["after_balancing"]) 
                     for i in range(5) if f"fold_{i}" in stats["folds"] and stats["folds"][f"fold_{i}"]["after_balancing"])
    efficiency = (total_after / total['total']) * 100 if total['total'] > 0 else 0
    
    print(f"\nData Usage Efficiency: {efficiency:.1f}% (after balancing)")

def main():
    parser = argparse.ArgumentParser(description="Analyze dataset statistics")
    parser.add_argument("stats_file", help="Path to dataset statistics JSON file")
    parser.add_argument("--latest", action="store_true", help="Find and analyze the latest stats file")
    args = parser.parse_args()
    
    if args.latest:
        # Find the latest stats file
        log_files = glob.glob("logs/dataset_stats_*.json")
        if not log_files:
            print("No statistics files found in logs/")
            return
        stats_file = max(log_files, key=os.path.getctime)
        print(f"Analyzing latest file: {stats_file}")
    else:
        stats_file = args.stats_file
    
    # Load and analyze statistics
    stats = load_stats(stats_file)
    
    print(f"\n=== Dataset Statistics Analysis ===")
    print(f"Signal: {stats['signal']}")
    print(f"Channel: {stats['channel']}")
    print(f"Mode: {'Multi-class' if stats['multiclass'] else 'Binary'} classification")
    if not stats['multiclass']:
        print(f"Background: {stats['background']}")
    print(f"Timestamp: {stats['timestamp']}")
    
    print_era_summary(stats)
    print_fold_summary(stats)
    print_total_summary(stats)

if __name__ == "__main__":
    main()
