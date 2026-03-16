#!/usr/bin/env python3
"""
Combine CnC datacards across channels and eras using combineCards.py.

Usage:
    combineDatacardsCnC.py --mode channel --era 2018 --masspoint MHc130_MA90 --method Baseline --binning uniform
    combineDatacardsCnC.py --mode era --channel Combined --masspoint MHc130_MA90 --method Baseline --binning uniform
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

# Era and channel definitions (same as combineDatacards.py)
RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]
CHANNELS = ["SR1E2Mu", "SR3Mu"]


def combine_channels(era, masspoint, method, binning, workdir, nsigma_tag="3sigma"):
    """Combine SR1E2Mu + SR3Mu -> Combined (CnC)."""
    output_dir = Path(f"{workdir}/SignalRegionStudyV2/templates/{era}/Combined/{masspoint}/{method}/{binning}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datacards = {}
    for channel in CHANNELS:
        card_path = f"{workdir}/SignalRegionStudyV2/templates/{era}/{channel}/{masspoint}/{method}/{binning}/datacard_cnc_{nsigma_tag}.txt"
        if Path(card_path).exists():
            datacards[channel] = card_path
        else:
            logging.warning(f"CnC datacard not found: {card_path}")

    if len(datacards) < 2:
        raise FileNotFoundError(f"Need both channels, found: {list(datacards.keys())}")

    cmd = ["combineCards.py"]
    for ch, path in datacards.items():
        cmd.append(f"{ch}={path}")

    logging.info(f"Combining CnC channels for {era}/{masspoint} ({nsigma_tag})")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"combineCards.py failed: {result.stderr}")

    output_card = output_dir / f"datacard_cnc_{nsigma_tag}.txt"
    with open(output_card, 'w') as f:
        f.write(result.stdout)

    logging.info(f"Created: {output_card}")
    return output_card


def combine_eras(eras, channel, masspoint, method, binning, output_era, workdir, nsigma_tag="3sigma"):
    """Combine multiple eras into one CnC datacard."""
    output_dir = Path(f"{workdir}/SignalRegionStudyV2/templates/{output_era}/{channel}/{masspoint}/{method}/{binning}")
    output_dir.mkdir(parents=True, exist_ok=True)

    datacards = {}
    for era in eras:
        card_path = Path(f"{workdir}/SignalRegionStudyV2/templates/{era}/{channel}/{masspoint}/{method}/{binning}/datacard_cnc_{nsigma_tag}.txt")
        if card_path.exists():
            datacards[f"era{era}"] = str(card_path)
        else:
            logging.warning(f"CnC datacard not found: {card_path}")

    if len(datacards) < 2:
        logging.warning(f"Only {len(datacards)} era(s) found, skipping combination")
        return None

    cmd = ["combineCards.py"]
    for era_label, path in datacards.items():
        cmd.append(f"{era_label}={path}")

    logging.info(f"Combining {len(datacards)} eras for {channel}/{masspoint} -> {output_era} ({nsigma_tag})")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"combineCards.py failed: {result.stderr}")

    output_card = output_dir / f"datacard_cnc_{nsigma_tag}.txt"
    with open(output_card, 'w') as f:
        f.write(result.stdout)

    logging.info(f"Created: {output_card}")
    return output_card


def main():
    if not shutil.which("combineCards.py"):
        logging.error(
            "combineCards.py not found in PATH. "
            "Please set up HiggsCombine (e.g., source setup.sh in a CMSSW environment with HiggsCombine installed)."
        )
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Combine CnC datacards across channels/eras")
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
    parser.add_argument("--unblind", action="store_true",
                        help="Use full unblind templates")
    parser.add_argument("--nsigma", type=float, default=3.0,
                        help="Mass window half-width in sigma_voigt (default: 3.0)")
    parser.add_argument("--output-era", default="Run2",
                        help="Output era name for era combination")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.unblind and args.partial_unblind:
        raise ValueError("--unblind and --partial-unblind are mutually exclusive")

    WORKDIR = os.getenv("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    binning_suffix = args.binning
    if args.unblind:
        binning_suffix = f"{args.binning}_unblind"
    elif args.partial_unblind:
        binning_suffix = f"{args.binning}_partial_unblind"

    nsigma_tag = f"{args.nsigma:g}sigma"

    try:
        if args.mode == "channel":
            if not args.era:
                raise ValueError("--era is required for channel combination")
            combine_channels(args.era, args.masspoint, args.method, binning_suffix, WORKDIR, nsigma_tag)

        elif args.mode == "era":
            if not args.channel:
                raise ValueError("--channel is required for era combination")
            eras = args.eras.split(",") if args.eras else RUN2_ERAS
            combine_eras(eras, args.channel, args.masspoint, args.method,
                        binning_suffix, args.output_era, WORKDIR, nsigma_tag)

    except Exception as e:
        logging.error(f"Failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
