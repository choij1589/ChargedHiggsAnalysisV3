#!/usr/bin/env python3
"""
Generate HiggsCombine datacards from shape templates.

This script reads the shapes.root file and systematics configuration to produce
a properly formatted datacard for limit extraction.

Usage:
    printDatacard.py --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended
"""
import os
import sys
import json
import logging
import argparse
import ROOT
from run_period_utils import is_signal_component

# Argument parsing
parser = argparse.ArgumentParser(description="Generate HiggsCombine datacard from templates")
parser.add_argument("--era", required=True, type=str, help="Run-period target: Run2, Run3, or All")
parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu, Combined)")
parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet)")
parser.add_argument("--binning", default="extended",
                    choices=["extended", "uniform"],
                    help="Binning method: 'extended' or 'uniform'")
parser.add_argument("--unblind", action="store_true",
                    help="Generate datacard from unblind run")
parser.add_argument("--partial-unblind", action="store_true", dest="partial_unblind",
                    help="Generate datacard from partial-unblind run")
parser.add_argument("--nuisance", default="fallback_lnn",
                    choices=["fallback_lnn", "preserve_shape"],
                    help="Low-stat nuisance handling: fallback_lnn keeps current shape?->lnN fallback; preserve_shape keeps shape variations")
parser.add_argument("--output", type=str, default=None, help="Output datacard path (default: auto-determined)")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

# Validate unblind options
if args.unblind and args.partial_unblind:
    raise ValueError("--unblind and --partial-unblind are mutually exclusive")

# Setup logging
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(levelname)s - %(message)s')

# Path setup
WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

# Template directory
binning_suffix = args.binning
if args.unblind:
    binning_suffix = f"{args.binning}_unblind"
elif args.partial_unblind:
    binning_suffix = f"{args.binning}_partial_unblind"
if args.nuisance == "preserve_shape":
    binning_suffix = f"{binning_suffix}_preserve_shape"
TEMPLATE_DIR = f"{WORKDIR}/SignalRegionStudyV3/templates/{args.era}/{args.channel}/{args.masspoint}/{args.method}/{binning_suffix}"

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
    config_path = f"{WORKDIR}/SignalRegionStudyV3/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    if channel not in config:
        raise ValueError(f"Channel '{channel}' not found in {config_path}")
    return config[channel]


class RunPeriodDatacardManager:
    """Generate datacards for Run-period categories with subera components."""

    def __init__(self, era, channel, masspoint, method, binning, nuisance_mode="fallback_lnn"):
        self.era = era
        self.channel = channel
        self.signal = masspoint
        self.method = method
        self.binning = binning
        self.nuisance_mode = nuisance_mode
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
        self._preserve_shape_invalid = {}

        self._load_systematics()
        self._build_columns()
        if not self.columns:
            raise ValueError("No active process columns found in run-period templates")

    def _load_systematics(self):
        for cat, payload in self.categories["categories"].items():
            channel = payload["channel"]
            for subera in payload["suberas"]:
                key = (subera, channel)
                if key not in self.systematics:
                    self.systematics[key] = load_systematics_block(subera, channel)

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

    def find_invalid_shape_pair(self, col, syst_name):
        cat = col["category"]
        proc = col["process"]
        nominal = self.get_hist_integral(cat, proc)
        up = self.get_hist_integral(cat, f"{proc}_{syst_name}Up")
        down = self.get_hist_integral(cat, f"{proc}_{syst_name}Down")
        if nominal is None or nominal <= MIN_YIELD_THRESHOLD:
            return None
        reasons = []
        if up is None:
            reasons.append("up_missing")
        elif up <= 0:
            reasons.append("up_integral <= 0")
        if down is None:
            reasons.append("down_missing")
        elif down <= 0:
            reasons.append("down_integral <= 0")
        if not reasons:
            return None
        fallback = self.compute_lnn_fallback_value(cat, proc, syst_name)
        return {
            "category": cat,
            "process": proc,
            "systematic": syst_name,
            "nominal_integral": nominal,
            "up_integral": up,
            "down_integral": down,
            "fallback": fallback,
            "reason": ", ".join(reasons),
        }

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

    def precompute_preserve_shape_invalid_fallbacks(self):
        for syst_name in self.all_systematic_names():
            for col in self.columns:
                syst_config = self.systematic_for_column(col, syst_name)
                if not syst_config or syst_config.get("type") != "shape":
                    continue
                if not self.process_applies(col, syst_config):
                    continue
                invalid = self.find_invalid_shape_pair(col, syst_name)
                if not invalid:
                    continue
                key = (col["category"], col["process"], syst_name)
                self._preserve_shape_invalid[key] = invalid
                self._lnn_fallback_cache[key] = invalid["fallback"]
                logging.warning(
                    "Preserve-shape fallback %s/%s/%s: %s",
                    col["category"], col["process"], syst_name, invalid["reason"]
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

    def rewrite_preserve_shape_invalid_hists(self):
        hists_to_remove = set()
        for cat, proc, syst_name in self._preserve_shape_invalid:
            hists_to_remove.add(f"{cat}/{proc}_{syst_name}Up")
            hists_to_remove.add(f"{cat}/{proc}_{syst_name}Down")
        self.rewrite_shapes_root_removing(hists_to_remove, "invalid preserve-shape")

    def write_lowstat_json(self):
        lowstat = [
            col for col in self.columns
            if (not col["is_signal"] and
                self.process_rel_errors.get((col["category"], col["process"]), float("inf")) > SHAPE_REL_ERR_THRESHOLD)
        ]
        if not lowstat and not self._preserve_shape_invalid:
            return
        fallbacks = {}
        for (cat, proc, syst), value in self._lnn_fallback_cache.items():
            fallbacks.setdefault(cat, {}).setdefault(proc, {})[syst] = value
        payload = {
            "construction": "run_period_components",
            "threshold": SHAPE_REL_ERR_THRESHOLD,
            "nuisance_mode": self.nuisance_mode,
            "processes": [
                {"category": col["category"], "process": col["process"]}
                for col in lowstat
            ],
            "fallbacks": fallbacks,
            "preserve_shape_invalid_fallbacks": list(self._preserve_shape_invalid.values()),
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
                if rel > SHAPE_REL_ERR_THRESHOLD and self.nuisance_mode == "fallback_lnn":
                    return self._lnn_fallback_cache.get(key, "-")
            if self.nuisance_mode == "preserve_shape" and key in self._preserve_shape_invalid:
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
            f"# Method: {self.method}, Binning: {self.binning}",
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
        if self.nuisance_mode == "fallback_lnn":
            self.precompute_lnn_fallbacks()
        else:
            self.precompute_preserve_shape_invalid_fallbacks()
        self.write_lowstat_json()
        if self.nuisance_mode == "fallback_lnn":
            self.rewrite_shapes_root()
        else:
            self.rewrite_preserve_shape_invalid_hists()
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
    logging.info(f"  Binning: {args.binning}")
    logging.info(f"  Template dir: {TEMPLATE_DIR}")

    categories = load_categories()
    if not categories:
        logging.error("Run-period metadata not found: %s/categories.json", TEMPLATE_DIR)
        logging.error("SignalRegionStudyV3 datacards require run_period_components templates.")
        sys.exit(1)

    logging.info("Detected run_period_components template metadata")
    manager = None
    try:
        manager = RunPeriodDatacardManager(
            args.era, args.channel, args.masspoint, args.method,
            args.binning, args.nuisance
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
