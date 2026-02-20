#!/usr/bin/env python3
"""
Combine datacards across channels and eras using combineCards.py.

Usage:
    combineDatacards.py --mode channel --era 2018 --masspoint MHc130_MA90 --method Baseline --binning uniform
    combineDatacards.py --mode era --channel Combined --masspoint MHc130_MA90 --method Baseline --binning uniform
"""
import os
import sys
import subprocess
import argparse
import logging
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Era and channel definitions
RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]
CHANNELS = ["SR1E2Mu", "SR3Mu"]


def combine_channels(era, masspoint, method, binning, workdir):
    """Combine SR1E2Mu + SR3Mu -> Combined."""
    output_dir = Path(f"{workdir}/SignalRegionStudyV1/templates/{era}/Combined/{masspoint}/{method}/{binning}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datacards = {}
    shapes_files = {}

    for channel in CHANNELS:
        card_path = f"{workdir}/SignalRegionStudyV1/templates/{era}/{channel}/{masspoint}/{method}/{binning}/datacard.txt"
        shapes_path = f"{workdir}/SignalRegionStudyV1/templates/{era}/{channel}/{masspoint}/{method}/{binning}/shapes.root"

        if Path(card_path).exists():
            datacards[channel] = card_path
            shapes_files[channel] = shapes_path
        else:
            logging.warning(f"Datacard not found: {card_path}")

    if len(datacards) < 2:
        raise FileNotFoundError(f"Need both channels, found: {list(datacards.keys())}")

    # Run combineCards.py
    cmd = ["combineCards.py"]
    for ch, path in datacards.items():
        cmd.append(f"{ch}={path}")

    logging.info(f"Combining channels for {era}/{masspoint}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"combineCards.py failed: {result.stderr}")

    # Copy shapes.root files to combined directory and fix paths to relative
    combined_content = result.stdout
    for channel, shapes_path in shapes_files.items():
        if Path(shapes_path).exists():
            dest_name = f"shapes_{channel}.root"
            dest_shapes = output_dir / dest_name
            shutil.copy(shapes_path, dest_shapes)
            # Replace absolute path with relative path in datacard
            combined_content = combined_content.replace(shapes_path, dest_name)

    # Write combined datacard with relative paths
    output_card = output_dir / "datacard.txt"
    with open(output_card, 'w') as f:
        f.write(combined_content)

    logging.info(f"Created: {output_card}")
    return output_card


def combine_eras(eras, channel, masspoint, method, binning, output_era, workdir):
    """Combine multiple eras into one datacard."""
    output_dir = Path(f"{workdir}/SignalRegionStudyV1/templates/{output_era}/{channel}/{masspoint}/{method}/{binning}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datacards = {}
    shapes_files = {}
    for era in eras:
        base_dir = Path(f"{workdir}/SignalRegionStudyV1/templates/{era}/{channel}/{masspoint}/{method}/{binning}")
        card_path = base_dir / "datacard.txt"
        if card_path.exists():
            datacards[f"era{era}"] = str(card_path)
            # Find all shapes files (could be shapes.root or shapes_*.root for Combined channel)
            for shapes_file in base_dir.glob("shapes*.root"):
                # shapes.root -> shapes_2018.root
                # shapes_SR1E2Mu.root -> shapes_2018_SR1E2Mu.root
                if shapes_file.name == "shapes.root":
                    dest_name = f"shapes_{era}.root"
                else:
                    # shapes_SR1E2Mu.root -> shapes_2018_SR1E2Mu.root
                    suffix = shapes_file.name.replace("shapes_", "").replace("shapes", "")
                    dest_name = f"shapes_{era}_{suffix}" if suffix else f"shapes_{era}.root"
                shapes_files[str(shapes_file)] = dest_name
        else:
            logging.warning(f"Datacard not found: {card_path}")

    if len(datacards) < 2:
        logging.warning(f"Only {len(datacards)} era(s) found, skipping combination")
        return None

    # Run combineCards.py
    cmd = ["combineCards.py"]
    for era_label, path in datacards.items():
        cmd.append(f"{era_label}={path}")

    logging.info(f"Combining {len(datacards)} eras for {channel}/{masspoint} -> {output_era}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"combineCards.py failed: {result.stderr}")

    # Copy shapes.root files to combined directory and fix paths to relative
    combined_content = result.stdout
    for src_path, dest_name in shapes_files.items():
        if Path(src_path).exists():
            dest_shapes = output_dir / dest_name
            shutil.copy(src_path, dest_shapes)
            # Replace absolute path with relative path in datacard
            combined_content = combined_content.replace(src_path, dest_name)

    # Write combined datacard with relative paths
    output_card = output_dir / "datacard.txt"
    with open(output_card, 'w') as f:
        f.write(combined_content)

    logging.info(f"Created: {output_card}")
    return output_card


def main():
    parser = argparse.ArgumentParser(description="Combine datacards across channels/eras")
    parser.add_argument("--mode", required=True, choices=["channel", "era"],
                        help="Combination mode: 'channel' or 'era'")
    parser.add_argument("--era", default=None,
                        help="Era for channel combination")
    parser.add_argument("--eras", default=None,
                        help="Comma-separated eras for era combination")
    parser.add_argument("--channel", default=None,
                        help="Channel for era combination")
    parser.add_argument("--masspoint", required=True,
                        help="Signal mass point")
    parser.add_argument("--method", required=True,
                        help="Template method")
    parser.add_argument("--binning", default="uniform",
                        help="Binning scheme")
    parser.add_argument("--partial-unblind", action="store_true", dest="partial_unblind",
                        help="Use partial-unblind templates (score < 0.3)")
    parser.add_argument("--output-era", default="FullRun2",
                        help="Output era name for era combination")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get WORKDIR
    WORKDIR = os.getenv("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    # Construct binning suffix
    binning_suffix = args.binning
    if args.partial_unblind:
        binning_suffix = f"{args.binning}_partial_unblind"

    try:
        if args.mode == "channel":
            if not args.era:
                raise ValueError("--era is required for channel combination")
            combine_channels(args.era, args.masspoint, args.method, binning_suffix, WORKDIR)

        elif args.mode == "era":
            if not args.channel:
                raise ValueError("--channel is required for era combination")
            eras = args.eras.split(",") if args.eras else RUN2_ERAS
            combine_eras(eras, args.channel, args.masspoint, args.method,
                        binning_suffix, args.output_era, WORKDIR)

    except Exception as e:
        logging.error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
