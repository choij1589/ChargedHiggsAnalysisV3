#!/usr/bin/env python3
"""Collect per-template ParticleNet background weights and thresholds into a
single config file per masspoint: configs/thresholds/{masspoint}.json.

Must be run after makeBinnedTemplates.py has produced templates for the desired
eras and channels. Reads from:
  templates/{era}/{channel}/{masspoint}/ParticleNet/{binning}/background_weights.json
  templates/{era}/{channel}/{masspoint}/ParticleNet/{binning}/threshold.json
"""

import argparse
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

# eras from automize/makeBinnedTemplates.sh
ERAS = ["2016preVFP", "2016postVFP", "2017", "2018",
        "2022", "2022EE", "2023", "2023BPix"]
CHANNELS = ["SR1E2Mu", "SR3Mu"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect ParticleNet thresholds and background weights into configs/thresholds/."
    )
    parser.add_argument(
        "--masspoint",
        help="Single masspoint to collect (default: all particlenet masspoints from configs/masspoints.json)"
    )
    parser.add_argument(
        "--binning", default="extended",
        help="Binning suffix to read from (default: extended). "
             "Avoid partial_unblind suffixes — those use a fixed upper_threshold, not an optimized one."
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_masspoints(workdir):
    path = f"{workdir}/SignalRegionStudyV3/configs/masspoints.json"
    with open(path) as f:
        cfg = json.load(f)
    return cfg["particlenet"]


def collect_one(workdir, masspoint, binning):
    """Collect all (era, channel) entries for one masspoint. Returns a dict."""
    result = {
        "masspoint": masspoint,
        "binning": binning,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_template_dir": f"templates/{{era}}/{{channel}}/{masspoint}/ParticleNet/{binning}",
    }

    n_populated = 0

    for era in ERAS:
        result.setdefault(era, {})
        for channel in CHANNELS:
            tdir = f"{workdir}/SignalRegionStudyV3/templates/{era}/{channel}/{masspoint}/ParticleNet/{binning}"
            bw_path = f"{tdir}/background_weights.json"
            thr_path = f"{tdir}/threshold.json"

            missing = []
            if not os.path.exists(bw_path):
                missing.append(bw_path)
            if not os.path.exists(thr_path):
                missing.append(thr_path)

            if missing:
                for p in missing:
                    logging.warning(f"Missing: {p}")
                result[era][channel] = None
                continue

            with open(bw_path) as f:
                bw = json.load(f)
            with open(thr_path) as f:
                thr = json.load(f)

            threshold = thr.get("threshold")
            if threshold is None:
                logging.warning(
                    f"No 'threshold' key in {thr_path} — "
                    f"this looks like a partial_unblind variant. Entry set to null."
                )
                result[era][channel] = None
                continue

            result[era][channel] = {
                "mass_window": bw["mass_window"],
                "weights": bw["weights"],
                "yields": bw["yields"],
                "total_yield": bw["total_yield"],
                "threshold": threshold,
                "sensitivity": {
                    "initial": thr.get("initial_sensitivity"),
                    "max": thr.get("max_sensitivity"),
                    "improvement": thr.get("improvement"),
                },
            }
            n_populated += 1

    logging.info(
        f"{masspoint}: {n_populated}/{len(ERAS) * len(CHANNELS)} (era, channel) entries populated"
    )
    return result, n_populated


def write_atomic(data, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_path.parent, suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.replace(tmp_path, out_path)
    except Exception:
        os.unlink(tmp_path)
        raise
    logging.info(f"Wrote {out_path}")


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh' first.")

    if args.masspoint:
        masspoints = [args.masspoint]
    else:
        masspoints = load_masspoints(workdir)

    for masspoint in masspoints:
        data, n = collect_one(workdir, masspoint, args.binning)
        out_path = f"{workdir}/SignalRegionStudyV3/configs/thresholds/{masspoint}.json"
        write_atomic(data, out_path)
        print(
            f"Wrote {out_path} — "
            f"{n}/{len(ERAS) * len(CHANNELS)} (era, channel) entries populated"
        )


if __name__ == "__main__":
    main()
