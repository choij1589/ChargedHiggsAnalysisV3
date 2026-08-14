#!/usr/bin/env python3
"""
Generate HiggsCombine datacards from shape templates.

This script reads the shapes.root file and systematics configuration to produce
a properly formatted datacard for limit extraction.

Usage:
    printDatacard.py --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline
"""
import os
import sys
import glob
import json
import logging
import argparse
import ROOT
from run_period_utils import is_signal_component
import srspaths

# Argument parsing
parser = argparse.ArgumentParser(description="Generate HiggsCombine datacard from templates")
parser.add_argument("--era", required=True, type=str, help="Run-period target: Run2, Run3, or All")
parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu, Combined)")
parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet)")
parser.add_argument("--blind", action="store_true",
                    help="Read from the {method}_blind template segment")
parser.add_argument("--signal-source", default="mc-signal",
                    choices=["mc-signal", "interp-signal"],
                    help="Template signal source (interp-signal members "
                         "resolve through their grid.json group seed)")
parser.add_argument("--output", type=str, default=None, help="Output datacard path (default: auto-determined)")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(levelname)s - %(message)s')

def _resolve_template_dir():
    if args.signal_source == "interp-signal":
        import interpolation_config
        seed = interpolation_config.group_seed(args.masspoint, args.method)
        if seed != args.masspoint:
            return srspaths.interp_member_dir(seed, args.masspoint,
                                              args.era, args.channel,
                                              blind=args.blind,
                                              method=args.method)
    return srspaths.template_dir(args.masspoint, args.method, args.era,
                                 args.channel, blind=args.blind,
                                 source=args.signal_source)


TEMPLATE_DIR = _resolve_template_dir()

# Setup ROOT
ROOT.gROOT.SetBatch(True)

# Threshold for using shape vs lnN systematics
# Backgrounds with relative statistical error above this threshold will use lnN fallback
# Relative error = sqrt(sum of bin_error^2) / integral
SHAPE_REL_ERR_THRESHOLD = 0.30  # 30%

# Constants for lnN fallback computation
MAX_LNN_VALUE = 2.0           # Cap lnN fallback (100% uncertainty max)
MIN_YIELD_THRESHOLD = 1e-6    # Below this yield, skip systematic entirely ("-")


def load_process_list():
    """Load the process list from template directory."""
    process_list_path = f"{TEMPLATE_DIR}/process_list.json"
    if not os.path.exists(process_list_path):
        raise FileNotFoundError(f"Process list not found: {process_list_path}")

    with open(process_list_path) as f:
        return json.load(f)


def load_categories():
    path = f"{TEMPLATE_DIR}/categories.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_systematics_block(era, channel):
    config = srspaths.systematics_config(era)
    if channel not in config:
        raise ValueError(f"Channel '{channel}' not found in systematics.{era}.json")
    return config[channel]


def load_extra_systematics():
    """Template-directory nuisances that no era config can carry.

    Optional: template producers that add their own nuisances (e.g. the
    interpolation uncertainties of a parametric-signal template) write
    extra_systematics*.json next to shapes.root, keyed "{subera}|{channel}".
    Several such files may be present — one per merged component directory —
    and are unioned. Absent for every production template: the datacard is
    then byte-for-byte what it always was.
    """
    merged = {}
    for path in sorted(glob.glob(f"{TEMPLATE_DIR}/extra_systematics*.json")):
        with open(path) as f:
            for key, block in json.load(f)["systematics"].items():
                merged.setdefault(key, {}).update(block)
    return merged


class RunPeriodDatacardManager:
    """Generate datacards for Run-period categories with subera components."""

    def __init__(self, era, channel, masspoint, method):
        self.era = era
        self.channel = channel
        self.signal = masspoint
        self.method = method
        self.categories = load_categories()
        if not self.categories:
            raise FileNotFoundError(f"categories.json not found in {TEMPLATE_DIR}")
        self.process_config = load_process_list()
        self.rtfile = None
        shapes_path = f"{TEMPLATE_DIR}/shapes.root"
        if not os.path.exists(shapes_path):
            raise FileNotFoundError(f"Template file not found: {shapes_path}")
        self.rtfile = ROOT.TFile.Open(shapes_path, "READ")

        self.systematics = {}
        self.columns = []
        self.process_yields = {}
        self.process_rel_errors = {}
        self._lnn_fallback_cache = {}

        self._load_systematics()
        self._build_columns()
        if not self.columns:
            raise ValueError("No active process columns found in run-period templates")

    def _load_systematics(self):
        extra = load_extra_systematics()
        for cat, payload in self.categories["categories"].items():
            channel = payload["channel"]
            for subera in payload["suberas"]:
                key = (subera, channel)
                if key not in self.systematics:
                    block = dict(load_systematics_block(subera, channel))
                    block.update(extra.get(f"{subera}|{channel}", {}))
                    self.systematics[key] = block
        if extra:
            logging.info("Merged %d template-local systematics block(s) from "
                         "extra_systematics.json", len(extra))

    def _build_columns(self):
        for cat, payload in self.categories["categories"].items():
            for meta in payload["processes"]:
                proc = meta["name"]
                rate = self.get_event_rate(cat, proc)
                if rate <= MIN_YIELD_THRESHOLD:
                    if meta.get("is_signal"):
                        logging.warning("Signal component %s/%s has non-positive yield %.4g", cat, proc, rate)
                    else:
                        logging.warning("Dropping non-positive background component %s/%s (%.4g)", cat, proc, rate)
                        continue
                col = {
                    "category": cat,
                    "process": proc,
                    "base_process": meta["base_process"],
                    "physics_group": meta.get("physics_group", meta["base_process"]),
                    "subera": meta["subera"],
                    "channel": payload["channel"],
                    "is_signal": bool(meta.get("is_signal", False)),
                }
                self.columns.append(col)
                key = (cat, proc)
                self.process_yields[key] = rate
                self.process_rel_errors[key] = self.get_relative_error(cat, proc)

    def get_hist(self, category, hist_name):
        return self.rtfile.Get(f"{category}/{hist_name}")

    def get_event_rate(self, category, process):
        hist = self.get_hist(category, process)
        if not hist:
            return 0.0
        return hist.Integral()

    def get_relative_error(self, category, process):
        hist = self.get_hist(category, process)
        if not hist:
            return float("inf")
        integral = hist.Integral()
        if integral <= 0:
            return float("inf")
        sum_err2 = 0.0
        for i in range(1, hist.GetNbinsX() + 1):
            sum_err2 += hist.GetBinError(i) ** 2
        return (sum_err2 ** 0.5) / integral

    def check_histogram_exists(self, category, hist_name):
        hist = self.get_hist(category, hist_name)
        return hist is not None

    def get_hist_integral(self, category, hist_name):
        hist = self.get_hist(category, hist_name)
        if not hist:
            return None
        return hist.Integral()

    def compute_lnn_fallback_value(self, category, process, syst_name):
        central = self.get_hist(category, process)
        up = self.get_hist(category, f"{process}_{syst_name}Up")
        down = self.get_hist(category, f"{process}_{syst_name}Down")
        if not central:
            return "-"
        central_int = central.Integral()
        if central_int < MIN_YIELD_THRESHOLD:
            return "-"

        effects = []
        if up:
            effects.append(abs(up.Integral() - central_int) / central_int)
        if down:
            effects.append(abs(down.Integral() - central_int) / central_int)
        if not effects:
            return "-"
        rate_effect = max(effects)
        if rate_effect < 0.001:
            return "-"
        lnn = min(1.0 + rate_effect, MAX_LNN_VALUE)
        return f"{lnn:.3f}"

    def systematic_for_column(self, col, syst_name):
        return self.systematics.get((col["subera"], col["channel"]), {}).get(syst_name)

    def process_applies(self, col, syst_config):
        group = syst_config.get("group", [])
        proc_check = "signal" if col["is_signal"] else col["base_process"]
        return proc_check in group

    def all_systematic_names(self):
        names = set()
        for config in self.systematics.values():
            names.update(config.keys())
        return sorted(names)

    def precompute_lnn_fallbacks(self):
        lowstat = [
            col for col in self.columns
            if (not col["is_signal"] and
                self.process_rel_errors.get((col["category"], col["process"]), float("inf")) > SHAPE_REL_ERR_THRESHOLD)
        ]
        if lowstat:
            logging.info("Low-stat component backgrounds:")
            for col in lowstat:
                rel = self.process_rel_errors[(col["category"], col["process"])]
                rate = self.process_yields[(col["category"], col["process"])]
                logging.info("  %s/%s: rel_err=%.1f%% yield=%.6f",
                             col["category"], col["process"], rel * 100.0, rate)

        for syst_name in self.all_systematic_names():
            for col in lowstat:
                syst_config = self.systematic_for_column(col, syst_name)
                if not syst_config or syst_config.get("type") != "shape":
                    continue
                if not self.process_applies(col, syst_config):
                    continue
                key = (col["category"], col["process"], syst_name)
                self._lnn_fallback_cache[key] = self.compute_lnn_fallback_value(
                    col["category"], col["process"], syst_name
                )

    def rewrite_shapes_root_removing(self, hists_to_remove, reason):
        if not hists_to_remove:
            return
        shapes_path = f"{TEMPLATE_DIR}/shapes.root"
        original_path = f"{TEMPLATE_DIR}/shapes_original.root"
        all_hists = {}
        for cat in self.categories["categories"]:
            directory = self.rtfile.Get(cat)
            if not directory:
                continue
            for key in directory.GetListOfKeys():
                name = key.GetName()
                full_name = f"{cat}/{name}"
                if full_name in hists_to_remove:
                    continue
                hist = directory.Get(name)
                if hist:
                    hist.SetDirectory(0)
                    all_hists[full_name] = hist

        self.rtfile.Close()
        if os.path.exists(original_path):
            # A pre-prune archive is already there, so shapes.root is itself a
            # pruned file from an earlier pass or re-run.  Renaming over the
            # archive would destroy the only unpruned copy, which the run-period
            # merge relies on; keep it and discard the pruned intermediate.
            os.remove(shapes_path)
        else:
            os.rename(shapes_path, original_path)
        outfile = ROOT.TFile.Open(shapes_path, "RECREATE")
        dirs = {}
        for full_name, hist in all_hists.items():
            cat, name = full_name.split("/", 1)
            if cat not in dirs:
                dirs[cat] = outfile.mkdir(cat)
            dirs[cat].cd()
            hist.Write(name)
            outfile.cd()
        outfile.Close()
        logging.info("Removed %d %s histograms from shapes.root", len(hists_to_remove), reason)
        self.rtfile = ROOT.TFile.Open(shapes_path, "READ")

    def rewrite_shapes_root(self):
        hists_to_remove = set()
        for col in self.columns:
            if col["is_signal"]:
                continue
            rel = self.process_rel_errors.get((col["category"], col["process"]), float("inf"))
            if rel <= SHAPE_REL_ERR_THRESHOLD:
                continue
            for syst_name in self.all_systematic_names():
                syst_config = self.systematic_for_column(col, syst_name)
                if not syst_config or syst_config.get("type") != "shape":
                    continue
                if not self.process_applies(col, syst_config):
                    continue
                hists_to_remove.add(f"{col['category']}/{col['process']}_{syst_name}Up")
                hists_to_remove.add(f"{col['category']}/{col['process']}_{syst_name}Down")
        self.rewrite_shapes_root_removing(hists_to_remove, "low-stat shape")

    def write_lowstat_json(self):
        lowstat = [
            col for col in self.columns
            if (not col["is_signal"] and
                self.process_rel_errors.get((col["category"], col["process"]), float("inf")) > SHAPE_REL_ERR_THRESHOLD)
        ]
        if not lowstat:
            return
        fallbacks = {}
        for (cat, proc, syst), value in self._lnn_fallback_cache.items():
            fallbacks.setdefault(cat, {}).setdefault(proc, {})[syst] = value
        # nuisance_mode and preserve_shape_invalid_fallbacks are kept as
        # literals for artifact compatibility with the frozen V3 outputs.
        payload = {
            "construction": "run_period_components",
            "threshold": SHAPE_REL_ERR_THRESHOLD,
            "nuisance_mode": "fallback_lnn",
            "processes": [
                {"category": col["category"], "process": col["process"]}
                for col in lowstat
            ],
            "fallbacks": fallbacks,
            "preserve_shape_invalid_fallbacks": [],
        }
        with open(f"{TEMPLATE_DIR}/lowstat.json", "w") as f:
            json.dump(payload, f, indent=2)

    def format_syst_value(self, col, syst_name, syst_config):
        if not syst_config or not self.process_applies(col, syst_config):
            return "-"
        syst_type = syst_config.get("type")
        if syst_type == "lnN":
            return f"{syst_config.get('value', 1.0):.3f}"
        if syst_type == "shape":
            key = (col["category"], col["process"], syst_name)
            if not col["is_signal"]:
                rel = self.process_rel_errors.get((col["category"], col["process"]), float("inf"))
                if rel > SHAPE_REL_ERR_THRESHOLD:
                    return self._lnn_fallback_cache.get(key, "-")
            up = f"{col['process']}_{syst_name}Up"
            down = f"{col['process']}_{syst_name}Down"
            if self.check_histogram_exists(col["category"], up) and self.check_histogram_exists(col["category"], down):
                return "1"
            return "-"
        return "-"

    def part1_header(self):
        lines = [
            "# Datacard for B2G-25-013",
            f"# Era: {self.era}, Channel: {self.channel}, Signal: {self.signal}",
            f"# Method: {self.method}, Binning: extended",
            "# Construction: run_period_components",
            "",
            f"imax {len(self.categories['categories'])} number of bins",
            "jmax * number of backgrounds",
            "kmax * number of nuisance parameters",
            "-" * 80,
            "shapes * * shapes.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC",
            "-" * 80,
        ]
        return "\n".join(lines)

    def part2_observation(self):
        cats = list(self.categories["categories"].keys())
        bins = "bin          " + "".join(f"{cat:<24}" for cat in cats)
        obs = "observation  "
        for cat in cats:
            hist = self.get_hist(cat, "data_obs")
            obs += f"{(hist.Integral() if hist else 0.0):<24.4f}"
        return "\n".join([bins, obs, "-" * 80])

    def part3_rates(self):
        signal_ids = {}
        next_signal_id = 0
        background_ids = {}
        next_background_id = 1

        bin_line = "bin          "
        proc_names = "process      "
        proc_indices = "process      "
        rate_line = "rate         "

        for col in self.columns:
            proc = col["process"]
            bin_line += f"{col['category']:<24}"
            proc_names += f"{proc:<24}"
            if col["is_signal"] or is_signal_component(proc):
                if proc not in signal_ids:
                    signal_ids[proc] = next_signal_id
                    next_signal_id -= 1
                proc_id = signal_ids[proc]
            else:
                base = col["base_process"]
                if base not in background_ids:
                    background_ids[base] = next_background_id
                    next_background_id += 1
                proc_id = background_ids[base]
            proc_indices += f"{proc_id:<24}"
            rate_line += f"{-1:<24}"
        return "\n".join([bin_line, proc_names, proc_indices, rate_line, "-" * 80])

    def generate_systematic_lines(self):
        lines = []
        for syst_name in self.all_systematic_names():
            configs = [self.systematic_for_column(col, syst_name) for col in self.columns]
            active_configs = [cfg for cfg in configs if cfg]
            if not active_configs:
                continue
            syst_type = "shape" if any(cfg.get("type") == "shape" for cfg in active_configs) else active_configs[0].get("type")
            card_type = "shape?" if syst_type == "shape" else syst_type
            values = []
            for col in self.columns:
                values.append(self.format_syst_value(col, syst_name, self.systematic_for_column(col, syst_name)))
            if all(v == "-" for v in values):
                continue
            line = f"{syst_name:<50} {card_type:<8}" + "".join(f"{v:<24}" for v in values)
            lines.append(line)
        return "\n".join(lines)

    def generate_automc_lines(self, threshold=5):
        return "\n".join(f"{cat} autoMCStats {threshold}" for cat in self.categories["categories"])

    def generate_datacard(self):
        self.precompute_lnn_fallbacks()
        self.write_lowstat_json()
        self.rewrite_shapes_root()
        return "\n".join([
            self.part1_header(),
            self.part2_observation(),
            self.part3_rates(),
            self.generate_systematic_lines(),
            self.generate_automc_lines(),
        ]) + "\n"

    def close(self):
        if self.rtfile:
            self.rtfile.Close()


def main():
    logging.info(f"Generating datacard for {args.masspoint}")
    logging.info(f"  Era: {args.era}")
    logging.info(f"  Channel: {args.channel}")
    logging.info(f"  Method: {args.method}")
    logging.info(f"  Template dir: {TEMPLATE_DIR}")

    categories = load_categories()
    if not categories:
        logging.error("Run-period metadata not found: %s/categories.json", TEMPLATE_DIR)
        logging.error("SignalRegionStudyV4 datacards require run_period_components templates.")
        sys.exit(1)

    logging.info("Detected run_period_components template metadata")
    manager = None
    try:
        manager = RunPeriodDatacardManager(
            args.era, args.channel, args.masspoint, args.method
        )
        datacard = manager.generate_datacard()
    except Exception as e:
        logging.error(f"Failed to generate run-period datacard: {e}")
        sys.exit(1)

    output_path = args.output or f"{TEMPLATE_DIR}/datacard.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(datacard)

    logging.info(f"Datacard saved to: {output_path}")
    logging.info("=" * 60)
    logging.info("Summary:")
    for cat in categories["categories"]:
        data_hist = manager.get_hist(cat, "data_obs")
        logging.info(f"  {cat}/data_obs: {(data_hist.Integral() if data_hist else 0.0):.4f}")
    logging.info(f"  Active columns: {len(manager.columns)}")
    manager.close()


if __name__ == "__main__":
    main()
