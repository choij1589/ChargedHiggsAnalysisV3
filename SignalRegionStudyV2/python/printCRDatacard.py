#!/usr/bin/env python3
"""
Generate a HiggsCombine datacard for the TTZ2E1Mu control region.

The datacard is structurally a standard signal+backgrounds card, but the signal
is a dummy 1e-6/bin placeholder written by makeCRTemplates.py — its only role
is to keep the `r` parameter alive so the existing partial-unblind GoF combine
command (`--freezeParameters r --setParameters r=0`) works without changes.
With r frozen at 0, the placeholder contributes nothing to the likelihood, so
the GoF / FitDiagnostics fits are physically background-only.

Systematics are pulled from the SR1E2Mu block of configs/systematics.{era}.json
(channel-correlated NPs are shared between SR and CR by design — the user
explicitly wants the same nonprompt norm, trigger SF, etc.).

Usage:
    python3 python/printCRDatacard.py --era 2018
"""
import os
import sys
import json
import logging
import argparse
import ROOT

# Make template_utils importable when running outside the SR python/ dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template_utils import SHAPE_REL_ERR_THRESHOLD


# Fixed parameters (mirror makeCRTemplates.py)
CHANNEL = "TTZ2E1Mu"
METHOD = "CR"
BINNING_TAG = "ZWin_adaptive"
DEFAULT_MASSPOINT = "MHc130_MA90"
SR_CONFIG_CHANNEL = "SR1E2Mu"

# Low-stat fallback: rate-effect bound (matches printDatacard.py)
MAX_LNN_VALUE = 2.0
MIN_YIELD_THRESHOLD = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", required=True, type=str)
    parser.add_argument("--masspoint", default=DEFAULT_MASSPOINT, type=str)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_systematics(workdir, era):
    config_path = f"{workdir}/SignalRegionStudyV2/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    if SR_CONFIG_CHANNEL not in cfg:
        raise ValueError(f"Channel '{SR_CONFIG_CHANNEL}' not found in {config_path}")
    return cfg[SR_CONFIG_CHANNEL]


def load_process_list(template_dir):
    path = f"{template_dir}/process_list.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"process_list.json not found: {path}")
    with open(path) as f:
        return json.load(f)


def get_event_rate(rfile, hist_name):
    h = rfile.Get(hist_name)
    return h.Integral() if h else 0.0


def get_relative_error(rfile, hist_name):
    h = rfile.Get(hist_name)
    if not h:
        return float("inf")
    integral = h.Integral()
    if integral <= 0:
        return float("inf")
    sum_err2 = sum(h.GetBinError(i) ** 2 for i in range(1, h.GetNbinsX() + 1))
    return (sum_err2 ** 0.5) / integral


def compute_lnn_fallback(rfile, process, syst_name):
    central = rfile.Get(process)
    up = rfile.Get(f"{process}_{syst_name}Up")
    dn = rfile.Get(f"{process}_{syst_name}Down")
    if not central:
        return "-"
    cint = central.Integral()
    if cint < MIN_YIELD_THRESHOLD:
        return "-"
    effects = []
    if up:
        effects.append(abs(up.Integral() - cint) / cint)
    if dn:
        effects.append(abs(dn.Integral() - cint) / cint)
    if not effects:
        return "-"
    rate_effect = max(effects)
    if rate_effect < 0.001:
        return "-"
    val = min(1.0 + rate_effect, MAX_LNN_VALUE)
    return f"{val:.3f}"


def precompute_lnn_fallbacks(rfile, lowstat_processes, syst_config):
    """Build {(proc, syst): lnN_string} for shape systs of low-stat processes."""
    fallbacks = {}
    for syst_name, props in syst_config.items():
        if props.get("type") != "shape":
            continue
        group = props.get("group", [])
        for proc in lowstat_processes:
            if proc not in group:
                continue
            fallbacks[(proc, syst_name)] = compute_lnn_fallback(rfile, proc, syst_name)
    return fallbacks


def rewrite_shapes_root_drop_lowstat(template_dir, lowstat_processes, syst_config):
    """Strip Up/Down shape histograms for low-stat processes so shape? falls back to lnN."""
    if not lowstat_processes:
        return

    shapes_path = f"{template_dir}/shapes.root"
    original_path = f"{template_dir}/shapes_original.root"

    to_remove = set()
    for syst_name, props in syst_config.items():
        if props.get("type") != "shape":
            continue
        group = props.get("group", [])
        for proc in lowstat_processes:
            if proc not in group:
                continue
            to_remove.add(f"{proc}_{syst_name}Up")
            to_remove.add(f"{proc}_{syst_name}Down")

    if not to_remove:
        return

    src = ROOT.TFile.Open(shapes_path, "READ")
    surviving = {}
    for key in src.GetListOfKeys():
        name = key.GetName()
        if name in to_remove:
            continue
        h = key.ReadObj()
        h.SetDirectory(0)
        surviving[name] = h
    src.Close()

    if os.path.exists(original_path):
        os.remove(original_path)
    os.rename(shapes_path, original_path)

    out = ROOT.TFile.Open(shapes_path, "RECREATE")
    for name, h in surviving.items():
        h.Write(name)
    out.Close()
    logging.info(f"Removed {len(to_remove)} low-stat shape hists; backup at shapes_original.root")


def format_syst_value(process, syst_name, syst_config_entry, rel_errors, fallback_cache,
                     rfile):
    group = syst_config_entry.get("group", [])
    source = syst_config_entry.get("source")
    syst_type = syst_config_entry.get("type")

    if process not in group:
        return "-"

    if syst_type == "lnN":
        return f"{syst_config_entry.get('value', 1.0):.3f}"

    if syst_type == "shape":
        is_background = process != "signal"
        if is_background and rel_errors.get(process, float("inf")) > SHAPE_REL_ERR_THRESHOLD:
            return fallback_cache.get((process, syst_name), "-")

        if source == "preprocessed":
            up_name = f"{process}_{syst_name}Up"
            return "1" if rfile.Get(up_name) else "-"
        return "1"

    return "-"


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s - %(message)s")

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'.")

    template_dir = (f"{workdir}/SignalRegionStudyV2/templates/{args.era}/{CHANNEL}/"
                    f"{args.masspoint}/{METHOD}/{BINNING_TAG}")
    if not os.path.isdir(template_dir):
        logging.error(f"Template directory not found: {template_dir}")
        sys.exit(1)

    syst_config = load_systematics(workdir, args.era)
    process_list = load_process_list(template_dir)
    separate_processes = process_list.get("separate_processes", [])
    backgrounds = list(separate_processes) + ["others"]
    processes = ["signal"] + backgrounds

    shapes_path = f"{template_dir}/shapes.root"
    rfile = ROOT.TFile.Open(shapes_path, "READ")
    if not rfile or rfile.IsZombie():
        logging.error(f"Could not open shapes.root: {shapes_path}")
        sys.exit(1)

    # Drop backgrounds with zero/negative integral (mirror SR behaviour)
    valid_backgrounds = []
    for bkg in backgrounds:
        rate = get_event_rate(rfile, bkg)
        if rate > 0:
            valid_backgrounds.append(bkg)
        else:
            logging.warning(f"Background '{bkg}' rate <= 0 ({rate:.6f}) — dropping from datacard")
    if not valid_backgrounds:
        logging.error("No backgrounds with positive yield; cannot build datacard.")
        sys.exit(1)
    processes = ["signal"] + valid_backgrounds

    rel_errors = {p: get_relative_error(rfile, p) for p in valid_backgrounds}
    rel_errors["signal"] = 0.0  # dummy

    # Recompute low-stat processes from current rates (lowstat.json was a hint)
    lowstat_processes = [
        p for p in valid_backgrounds
        if rel_errors[p] > SHAPE_REL_ERR_THRESHOLD
    ]
    if lowstat_processes:
        logging.info(f"Low-stat backgrounds (rel_err > {SHAPE_REL_ERR_THRESHOLD*100:.0f}%): {lowstat_processes}")
        for p in lowstat_processes:
            logging.info(f"  {p}: rel_err = {rel_errors[p]*100:.1f}%, yield = {get_event_rate(rfile, p):.4f}")

    # Pre-compute lnN fallbacks while Up/Down hists still exist; cache rates
    # before close so we don't need to reopen just to read them.
    fallback_cache = precompute_lnn_fallbacks(rfile, lowstat_processes, syst_config)
    obs = get_event_rate(rfile, "data_obs")
    signal_rate = get_event_rate(rfile, "signal")
    rfile.Close()

    lowstat_path = f"{template_dir}/lowstat.json"
    fallbacks_dict = {}
    for (proc, syst_name), val in fallback_cache.items():
        fallbacks_dict.setdefault(proc, {})[syst_name] = val
    with open(lowstat_path, "w") as f:
        json.dump({
            "threshold": SHAPE_REL_ERR_THRESHOLD,
            "processes": lowstat_processes,
            "fallbacks": fallbacks_dict,
        }, f, indent=2)

    # Strip Up/Down hists for low-stat processes; format_syst_value below uses the
    # rewritten file to detect surviving shape variations.
    rewrite_shapes_root_drop_lowstat(template_dir, lowstat_processes, syst_config)
    rfile = ROOT.TFile.Open(shapes_path, "READ")

    # =========================================================================
    # Build datacard text
    # =========================================================================
    lines = []
    lines.append(f"# TTZ2E1Mu CR datacard for B2G-25-013")
    lines.append(f"# Era: {args.era}, Channel: {CHANNEL}, Method: {METHOD}, Binning: {BINNING_TAG}")
    lines.append(f"# Background-only fit (signal is a 1e-6/bin placeholder, r frozen to 0)")
    lines.append("")
    lines.append(f"imax 1 number of bins")
    lines.append(f"jmax {len(valid_backgrounds)} number of backgrounds")
    lines.append(f"kmax * number of nuisance parameters")
    lines.append("-" * 80)
    lines.append(f"shapes * * shapes.root $PROCESS $PROCESS_$SYSTEMATIC")
    lines.append("-" * 80)
    lines.append(f"bin          {CHANNEL}")
    lines.append(f"observation  {obs:.4f}")
    lines.append("-" * 80)

    # Process rate block
    nproc = len(processes)
    bin_line = "bin                                                " + (f"{CHANNEL:<15}" * nproc)
    proc_names = "process                                            " + "".join(f"{p:<15}" for p in processes)
    proc_indices = "process                                            " + "".join(f"{i:<15}" for i in range(nproc))
    rate_line = "rate                                               " + ("-1             " * nproc)
    lines += [bin_line, proc_names, proc_indices, rate_line]
    lines.append("-" * 80)

    # Systematics block
    for syst_name, props in syst_config.items():
        syst_type = props.get("type")
        group = props.get("group", [])
        if not any(p in group for p in processes):
            continue

        if syst_type == "shape":
            values = [
                format_syst_value(p, syst_name, props, rel_errors, fallback_cache, rfile)
                for p in processes
            ]
            if all(v == "-" for v in values):
                continue
            line = f"{syst_name:<50} {'shape?':<8}" + "".join(f"{v:<15}" for v in values)
            lines.append(line)
        elif syst_type == "lnN":
            values = []
            for p in processes:
                if p not in group:
                    values.append("-")
                else:
                    values.append(f"{props.get('value', 1.0):.3f}")
            if all(v == "-" for v in values):
                continue
            line = f"{syst_name:<50} {'lnN':<8}" + "".join(f"{v:<15}" for v in values)
            lines.append(line)

    # autoMCStats — same threshold as SR
    lines.append(f"{CHANNEL} autoMCStats 5")

    rfile.Close()

    output = "\n".join(lines) + "\n"
    out_path = f"{template_dir}/datacard.txt"
    with open(out_path, "w") as f:
        f.write(output)

    logging.info("=" * 60)
    logging.info(f"Datacard written: {out_path}")
    logging.info(f"  Observation: {obs:.4f}")
    logging.info(f"  Signal (placeholder): {signal_rate:.4e}")
    logging.info(f"  Backgrounds: {valid_backgrounds}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
