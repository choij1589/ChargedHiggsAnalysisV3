#!/usr/bin/env python3
"""Collect optimized ParticleNet LR thresholds from produced templates.

Must be run after makeBinnedTemplates.py has produced templates. The collector
scans:
  templates/{era}/{channel}/{masspoint}/ParticleNet/{binning}/threshold.json

and writes one aggregate config per mass point:
  configs/thresholds/{masspoint}.json

Only optimized threshold entries are collected. Fixed partial-unblind
``upper_threshold`` entries are skipped.
"""

import argparse
import json
import logging
import os
import tempfile
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect optimized ParticleNet thresholds from templates into configs/thresholds/."
    )
    parser.add_argument(
        "--masspoint", action="append",
        help="Mass point to collect. Can be passed multiple times. Default: discover all produced ParticleNet templates."
    )
    parser.add_argument(
        "--binning", action="append",
        help="Binning suffix to collect. Can be passed multiple times. Default: all produced binnings."
    )
    parser.add_argument(
        "--template-root", default=str(MODULE_DIR / "templates"),
        help="Template root to scan (default: templates)."
    )
    parser.add_argument(
        "--output-dir", default=str(MODULE_DIR / "configs" / "thresholds"),
        help="Output directory for aggregate JSON files (default: configs/thresholds)."
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def read_json(path):
    with open(path) as f:
        return json.load(f)


def relative_to_module(path):
    path = Path(path)
    try:
        return str(path.relative_to(MODULE_DIR))
    except ValueError:
        return str(path)


def sensitivity_from(payload):
    return {
        "initial": payload.get("initial_sensitivity"),
        "max": payload.get("max_sensitivity"),
        "improvement": payload.get("improvement"),
    }


def add_background_info(entry, bg_payload):
    if not bg_payload:
        return entry
    for key in ("mass_window", "weights", "yields", "total_yield"):
        if key in bg_payload:
            entry[key] = bg_payload[key]
    return entry


def category_background_payload(bg_data, category):
    if not bg_data:
        return None
    categories = bg_data.get("categories")
    if isinstance(categories, dict):
        return categories.get(category)
    return bg_data


def normalize_thresholds(threshold_data, background_data):
    """Return normalized optimized threshold payload and number of thresholds."""
    construction = threshold_data.get("construction")
    categories = threshold_data.get("categories")

    if isinstance(categories, dict):
        normalized_categories = OrderedDict()
        for category in sorted(categories):
            payload = categories[category]
            threshold = payload.get("threshold")
            if threshold is None:
                continue
            entry = OrderedDict()
            entry["category"] = payload.get("category", category)
            entry["threshold"] = threshold
            entry["sensitivity"] = sensitivity_from(payload)
            add_background_info(entry, category_background_payload(background_data, category))
            normalized_categories[category] = entry

        if not normalized_categories:
            return None, 0

        result = OrderedDict()
        if construction:
            result["construction"] = construction
        result["categories"] = normalized_categories
        return result, len(normalized_categories)

    threshold = threshold_data.get("threshold")
    if threshold is None:
        return None, 0

    result = OrderedDict()
    if construction:
        result["construction"] = construction
    result["threshold"] = threshold
    result["sensitivity"] = sensitivity_from(threshold_data)
    add_background_info(result, background_data)
    return result, 1


def discover_threshold_files(template_root, masspoints=None, binnings=None):
    template_root = Path(template_root)
    masspoints = set(masspoints or [])
    binnings = set(binnings or [])

    # Expected V3 template depth:
    # templates/{era}/{channel}/{masspoint}/ParticleNet/{binning}/threshold.json
    for threshold_path in sorted(template_root.glob("*/*/*/ParticleNet/*/threshold.json")):
        try:
            rel = threshold_path.relative_to(template_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 6 or parts[3] != "ParticleNet" or parts[5] != "threshold.json":
            continue

        era, channel, masspoint, _, binning, _ = parts
        if masspoints and masspoint not in masspoints:
            continue
        if binnings and binning not in binnings:
            continue
        yield era, channel, masspoint, binning, threshold_path


def collect(template_root, masspoints=None, binnings=None):
    collected = defaultdict(lambda: OrderedDict())
    counts = defaultdict(int)
    seen_files = 0
    skipped_files = 0

    for era, channel, masspoint, binning, threshold_path in discover_threshold_files(
        template_root, masspoints=masspoints, binnings=binnings
    ):
        seen_files += 1
        template_dir = threshold_path.parent
        background_path = template_dir / "background_weights.json"

        threshold_data = read_json(threshold_path)
        background_data = read_json(background_path) if background_path.exists() else None
        template_payload, n_thresholds = normalize_thresholds(threshold_data, background_data)
        if not template_payload:
            skipped_files += 1
            logging.debug("Skipping non-optimized threshold file: %s", threshold_path)
            continue

        template_payload["source_dir"] = relative_to_module(template_dir)
        if not background_path.exists():
            logging.warning("Missing background weights for %s", relative_to_module(template_dir))

        masspoint_payload = collected[masspoint]
        masspoint_payload.setdefault(era, OrderedDict())
        masspoint_payload[era].setdefault(channel, OrderedDict())
        masspoint_payload[era][channel][binning] = template_payload
        counts[masspoint] += n_thresholds

    logging.info(
        "Scanned %d threshold files; skipped %d fixed/non-optimized files",
        seen_files, skipped_files,
    )
    return collected, counts


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

    template_root = Path(args.template_root)
    output_dir = Path(args.output_dir)
    if not template_root.exists():
        raise FileNotFoundError(f"Template root not found: {template_root}")

    collected, counts = collect(
        template_root,
        masspoints=args.masspoint,
        binnings=args.binning,
    )
    if not collected:
        logging.warning("No optimized ParticleNet thresholds found")
        return

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for masspoint in sorted(collected):
        data = OrderedDict()
        data["masspoint"] = masspoint
        data["generated_at"] = generated_at
        data["source_template_pattern"] = "templates/{era}/{channel}/{masspoint}/ParticleNet/{binning}"
        data["templates"] = collected[masspoint]
        out_path = output_dir / f"{masspoint}.json"
        write_atomic(data, out_path)
        print(
            f"Wrote {out_path} - "
            f"{counts[masspoint]} optimized thresholds from produced templates"
        )


if __name__ == "__main__":
    main()
