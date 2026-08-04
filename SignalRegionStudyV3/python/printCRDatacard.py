#!/usr/bin/env python3
"""Generate a Run-period component datacard for the TTZ2E1Mu control region."""

import argparse
import json
import logging
import os
import sys

import ROOT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_utils import SHAPE_REL_ERR_THRESHOLD


CHANNEL = "TTZ2E1Mu"
SYSTEMATICS_CHANNEL = "SR1E2Mu"
METHOD = "CR"
BINNING_TAG = "ZWin_adaptive"
DEFAULT_MASSPOINT = "MHc130_MA90"
MAX_LNN_VALUE = 2.0
MIN_YIELD_THRESHOLD = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era", required=True, help="Run-period target: Run2, Run3, or All")
    parser.add_argument("--masspoint", default=DEFAULT_MASSPOINT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def template_dir(workdir, era, masspoint):
    return f"{workdir}/SignalRegionStudyV3/templates/{era}/{CHANNEL}/{masspoint}/{METHOD}/{BINNING_TAG}"


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def load_systematics(workdir, subera):
    path = f"{workdir}/SignalRegionStudyV3/configs/systematics.{subera}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Systematics config not found: {path}")
    with open(path) as f:
        config = json.load(f)
    if SYSTEMATICS_CHANNEL not in config:
        raise ValueError(f"Channel '{SYSTEMATICS_CHANNEL}' not found in {path}")
    return config[SYSTEMATICS_CHANNEL]


class CRRunPeriodDatacard:
    def __init__(self, workdir, era, masspoint):
        self.workdir = workdir
        self.era = era
        self.masspoint = masspoint
        self.template_dir = template_dir(workdir, era, masspoint)
        self.categories = load_json(f"{self.template_dir}/categories.json")["categories"]
        self.process_list = load_json(f"{self.template_dir}/process_list.json")
        shapes_path = f"{self.template_dir}/shapes.root"
        if not os.path.exists(shapes_path):
            raise FileNotFoundError(shapes_path)
        self.rfile = ROOT.TFile.Open(shapes_path, "READ")
        if not self.rfile or self.rfile.IsZombie():
            raise OSError(f"Failed to open {shapes_path}")

        self.systematics = {}
        self.columns = []
        self.process_yields = {}
        self.process_rel_errors = {}
        self._lnn_fallback_cache = {}

        self._load_systematics()
        self._build_columns()
        if not self.columns:
            raise ValueError("No active process columns found in CR templates")

    def _load_systematics(self):
        for payload in self.categories.values():
            for subera in payload["suberas"]:
                if subera not in self.systematics:
                    self.systematics[subera] = load_systematics(self.workdir, subera)

    def _build_columns(self):
        for cat, payload in self.categories.items():
            for meta in payload["processes"]:
                proc = meta["name"]
                rate = self.get_event_rate(cat, proc)
                if rate <= MIN_YIELD_THRESHOLD and not meta.get("is_signal"):
                    logging.warning("Dropping non-positive CR component %s/%s (%.4g)", cat, proc, rate)
                    continue
                col = {
                    "category": cat,
                    "process": proc,
                    "base_process": meta["base_process"],
                    "physics_group": meta.get("physics_group", meta["base_process"]),
                    "subera": meta["subera"],
                    "is_signal": bool(meta.get("is_signal", False)),
                    "dummy_signal": bool(meta.get("dummy_signal", False)),
                }
                self.columns.append(col)
                key = (cat, proc)
                self.process_yields[key] = rate
                self.process_rel_errors[key] = self.get_relative_error(cat, proc)

    def get_hist(self, category, hist_name):
        return self.rfile.Get(f"{category}/{hist_name}")

    def get_event_rate(self, category, process):
        hist = self.get_hist(category, process)
        return hist.Integral() if hist else 0.0

    def get_relative_error(self, category, process):
        hist = self.get_hist(category, process)
        if not hist:
            return float("inf")
        integral = hist.Integral()
        if integral <= 0:
            return float("inf")
        sum_err2 = sum(hist.GetBinError(i) ** 2 for i in range(1, hist.GetNbinsX() + 1))
        return (sum_err2 ** 0.5) / integral

    def systematic_for_column(self, col, syst_name):
        return self.systematics.get(col["subera"], {}).get(syst_name)

    def process_applies(self, col, syst_config):
        if col["is_signal"]:
            return False
        return col["base_process"] in syst_config.get("group", [])

    def all_systematic_names(self):
        names = set()
        for config in self.systematics.values():
            names.update(config.keys())
        return sorted(names)

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
        return f"{min(1.0 + rate_effect, MAX_LNN_VALUE):.3f}"

    def precompute_lnn_fallbacks(self):
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
                key = (col["category"], col["process"], syst_name)
                self._lnn_fallback_cache[key] = self.compute_lnn_fallback_value(
                    col["category"], col["process"], syst_name
                )

    def rewrite_shapes_root(self):
        to_remove = set()
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
                if self.process_applies(col, syst_config):
                    to_remove.add(f"{col['category']}/{col['process']}_{syst_name}Up")
                    to_remove.add(f"{col['category']}/{col['process']}_{syst_name}Down")
        if not to_remove:
            return

        shapes_path = f"{self.template_dir}/shapes.root"
        original_path = f"{self.template_dir}/shapes_original.root"
        surviving = {}
        for cat in self.categories:
            directory = self.rfile.Get(cat)
            if not directory:
                continue
            for key in directory.GetListOfKeys():
                name = key.GetName()
                full_name = f"{cat}/{name}"
                if full_name in to_remove:
                    continue
                hist = directory.Get(name)
                if hist:
                    hist.SetDirectory(0)
                    surviving[full_name] = hist

        self.rfile.Close()
        if os.path.exists(original_path):
            # Already-pruned shapes.root from an earlier pass or re-run: keep
            # the existing pre-prune archive rather than overwriting it.
            os.remove(shapes_path)
        else:
            os.rename(shapes_path, original_path)

        out = ROOT.TFile.Open(shapes_path, "RECREATE")
        dirs = {}
        for full_name, hist in surviving.items():
            cat, name = full_name.split("/", 1)
            if cat not in dirs:
                dirs[cat] = out.mkdir(cat)
            dirs[cat].cd()
            hist.Write(name)
            out.cd()
        out.Close()
        logging.info("Removed %d low-stat CR shape histograms from shapes.root", len(to_remove))
        self.rfile = ROOT.TFile.Open(shapes_path, "READ")

    def write_lowstat_json(self):
        lowstat = [
            col for col in self.columns
            if not col["is_signal"] and
            self.process_rel_errors.get((col["category"], col["process"]), float("inf")) > SHAPE_REL_ERR_THRESHOLD
        ]
        fallbacks = {}
        for (cat, proc, syst_name), value in self._lnn_fallback_cache.items():
            fallbacks.setdefault(cat, {}).setdefault(proc, {})[syst_name] = value
        payload = {
            "construction": "run_period_components",
            "analysis": "TTZ2E1Mu_CR",
            "threshold": SHAPE_REL_ERR_THRESHOLD,
            "processes": [
                {"category": col["category"], "process": col["process"]}
                for col in lowstat
            ],
            "fallbacks": fallbacks,
        }
        with open(f"{self.template_dir}/lowstat.json", "w") as f:
            json.dump(payload, f, indent=2)

    def format_syst_value(self, col, syst_name, syst_config):
        if not syst_config or not self.process_applies(col, syst_config):
            return "-"
        syst_type = syst_config.get("type")
        if syst_type == "lnN":
            return f"{syst_config.get('value', 1.0):.3f}"
        if syst_type == "shape":
            key = (col["category"], col["process"], syst_name)
            rel = self.process_rel_errors.get((col["category"], col["process"]), float("inf"))
            if rel > SHAPE_REL_ERR_THRESHOLD:
                return self._lnn_fallback_cache.get(key, "-")
            up = self.get_hist(col["category"], f"{col['process']}_{syst_name}Up")
            down = self.get_hist(col["category"], f"{col['process']}_{syst_name}Down")
            return "1" if up and down else "-"
        return "-"

    def part1_header(self):
        return "\n".join([
            "# TTZ2E1Mu CR datacard for B2G-25-013",
            f"# Era: {self.era}, Channel: {CHANNEL}, Method: {METHOD}, Binning: {BINNING_TAG}",
            "# Construction: run_period_components",
            "# Background-only fit; signal components are 1e-6/bin placeholders and r is frozen to 0 in GoF.",
            "",
            f"imax {len(self.categories)} number of bins",
            "jmax * number of backgrounds",
            "kmax * number of nuisance parameters",
            "-" * 80,
            "shapes * * shapes.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC",
            "-" * 80,
        ])

    def part2_observation(self):
        cats = list(self.categories.keys())
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
            if col["is_signal"]:
                signal_ids.setdefault(proc, next_signal_id)
                proc_id = signal_ids[proc]
                if proc_id == next_signal_id:
                    next_signal_id -= 1
            else:
                base = col["base_process"]
                background_ids.setdefault(base, next_background_id)
                proc_id = background_ids[base]
                if proc_id == next_background_id:
                    next_background_id += 1
            proc_indices += f"{proc_id:<24}"
            rate_line += f"{-1:<24}"
        return "\n".join([bin_line, proc_names, proc_indices, rate_line, "-" * 80])

    def systematic_lines(self):
        lines = []
        for syst_name in self.all_systematic_names():
            active_configs = [
                self.systematic_for_column(col, syst_name)
                for col in self.columns
                if self.systematic_for_column(col, syst_name)
            ]
            if not active_configs:
                continue
            syst_type = "shape" if any(cfg.get("type") == "shape" for cfg in active_configs) else active_configs[0].get("type")
            card_type = "shape?" if syst_type == "shape" else syst_type
            values = [
                self.format_syst_value(col, syst_name, self.systematic_for_column(col, syst_name))
                for col in self.columns
            ]
            if all(value == "-" for value in values):
                continue
            lines.append(f"{syst_name:<50} {card_type:<8}" + "".join(f"{value:<24}" for value in values))
        return "\n".join(lines)

    def automc_lines(self):
        return "\n".join(f"{cat} autoMCStats 5" for cat in self.categories)

    def generate(self):
        self.precompute_lnn_fallbacks()
        self.write_lowstat_json()
        self.rewrite_shapes_root()
        return "\n".join([
            self.part1_header(),
            self.part2_observation(),
            self.part3_rates(),
            self.systematic_lines(),
            self.automc_lines(),
        ]) + "\n"

    def close(self):
        if self.rfile:
            self.rfile.Close()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'.")

    manager = None
    try:
        manager = CRRunPeriodDatacard(workdir, args.era, args.masspoint)
        datacard = manager.generate()
        out_path = f"{manager.template_dir}/datacard.txt"
        with open(out_path, "w") as f:
            f.write(datacard)
        logging.info("CR datacard written: %s", out_path)
        logging.info("Active columns: %d", len(manager.columns))
    except Exception as exc:
        logging.error("Failed to generate CR datacard: %s", exc)
        sys.exit(1)
    finally:
        if manager:
            manager.close()


if __name__ == "__main__":
    main()
