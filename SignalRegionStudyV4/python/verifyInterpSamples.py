#!/usr/bin/env python3
"""pnfs anti-truncation gate for the interpolation chain (shared sample
layout).

Concurrent preprocessing has previously lost or truncated pnfs files
silently, so every signal file must be opened and checked before any fit
runs. For each masspoint x era x shared channel dir this verifies the
signal file {masspoint}.root: exists, size > 0, opens cleanly, not
IsZombie, not kRecovered (a recovered file IS a truncation symptom ->
FAIL), has a 'Central' tree with entries > 0 and sum(weight) > 0. The
shared background/data files are checked for existence only (they are
mass-independent and were validated by the shared-layout refactor).

  python3 verifyInterpSamples.py --masspoints MHc145_MA15,MHc145_MA35
  python3 verifyInterpSamples.py --all --mhc 160
"""
import argparse
import os
import subprocess
import sys

import ROOT

import interpolation_config
import run_period_utils
import srspaths

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError


def pnfs_samples_base():
    """Site constant from scripts/env.sh (the single definition site),
    respecting an environment override."""
    env_sh = os.path.join(srspaths.module_dir(), "scripts", "env.sh")
    base = subprocess.check_output(
        ["bash", "-c", f'source "{env_sh}" && echo -n "$PNFS_USER_BASE"'],
        text=True)
    return os.path.join(base, srspaths.MODULE_NAME, "samples")


ERAS = run_period_utils.RUN2_ERAS + run_period_utils.RUN3_ERAS
SHARED_CHANNEL_DIRS = [
    srspaths.shared_channel_dirname("SR3Mu", pairing=ch.split("_", 1)[1])
    if ch.startswith("SR3Mu_") else ch
    for ch in interpolation_config.STUDY_CHANNELS
]
BACKGROUND_FILES = [f"{p}.root"
                    for p in run_period_utils.PHYSICS_PROCESS_ORDER
                    if p != "signal"] + ["data.root"]


def check_signal_file(path):
    """Return a list of failure strings for one signal file (empty = OK)."""
    if not os.path.exists(path):
        return [f"MISSING {path}"]
    if os.path.getsize(path) == 0:
        return [f"CORRUPT {path} : zero size"]

    fails = []
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        if f:
            f.Close()
        return [f"CORRUPT {path} : zombie / cannot open"]
    try:
        if f.TestBit(ROOT.TFile.kRecovered):
            fails.append(f"CORRUPT {path} : recovered file (truncation symptom)")
        tree = f.Get("Central")
        if not tree or not isinstance(tree, ROOT.TTree):
            fails.append(f"CORRUPT {path} : no 'Central' tree")
        elif tree.GetEntries() <= 0:
            fails.append(f"CORRUPT {path} : signal Central tree is empty")
        else:
            hname = "h_sumw_check"
            hist = ROOT.TH1D(hname, "", 1, -1e9, 1e9)
            tree.Draw(f"weight>>{hname}", "weight", "goff")
            sumw = hist.GetBinContent(1) + hist.GetBinContent(0) \
                + hist.GetBinContent(2)
            hist.Delete()
            if not sumw > 0:
                fails.append(f"CORRUPT {path} : signal sum(weight)={sumw}")
    finally:
        f.Close()
    return fails


def verify_masspoint(pnfs_samples, masspoint, checked_shared):
    fails = []
    for era in ERAS:
        for chdir in SHARED_CHANNEL_DIRS:
            sdir = os.path.join(pnfs_samples, era, chdir)
            if not os.path.isdir(sdir):
                fails.append(f"MISSING {sdir} : shared dir absent")
                continue
            key = (era, chdir)
            if key not in checked_shared:
                checked_shared.add(key)
                for bkg in BACKGROUND_FILES:
                    bpath = os.path.join(sdir, bkg)
                    if not os.path.exists(bpath) or os.path.getsize(bpath) == 0:
                        fails.append(f"MISSING {bpath} : shared background")
            fails.extend(check_signal_file(
                os.path.join(sdir, f"{masspoint}.root")))
    return fails


PNET_CHANNELS_WITH_SIGNAL = ("SR1E2Mu", "SR3Mu")
PNET_CHANNELS = PNET_CHANNELS_WITH_SIGNAL + ("TTZ2E1Mu",)


def verify_pnet_mhc(pnfs_samples, mhc):
    """Anti-truncation gate for the per-mHc shared-scores layout
    (preprocess.py --shared-scores): samples/{era}/{channel}/MHc{X}/ with
    every trained mass point's signal file (full check) and the shared
    backgrounds (existence). TTZ2E1Mu dirs carry backgrounds only. A dir
    holding a CENTRAL_ONLY marker is a study artifact, never a
    template-production input -> FAIL."""
    import pnet_interp_config as pic
    masspoints = pic.trained_masspoints(mhc)
    fails = []
    for era in ERAS:
        for channel in PNET_CHANNELS:
            sdir = os.path.join(pnfs_samples, era, channel, f"MHc{mhc}")
            if not os.path.isdir(sdir):
                fails.append(f"MISSING {sdir} : shared-scores dir absent")
                continue
            if os.path.exists(os.path.join(sdir, "CENTRAL_ONLY")):
                fails.append(f"CORRUPT {sdir} : CENTRAL_ONLY marker "
                             "(study output, not a production input)")
            for bkg in BACKGROUND_FILES:
                bpath = os.path.join(sdir, bkg)
                if not os.path.exists(bpath) or os.path.getsize(bpath) == 0:
                    fails.append(f"MISSING {bpath} : shared background")
            if channel in PNET_CHANNELS_WITH_SIGNAL:
                for mp in masspoints:
                    fails.extend(check_signal_file(
                        os.path.join(sdir, f"{mp}.root")))
    return fails


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study for --all / --pnet")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--masspoints", help="comma-separated masspoint names")
    group.add_argument("--all", action="store_true",
                       help="verify the full grid of the --mhc study")
    group.add_argument("--pnet", action="store_true",
                       help="verify the per-mHc shared-scores dirs "
                            "(ParticleNet interpolation inputs)")
    args = parser.parse_args()

    pnfs_samples = pnfs_samples_base()

    if args.pnet:
        fails = verify_pnet_mhc(pnfs_samples, args.mhc)
        status = "OK" if not fails else f"FAIL ({len(fails)} problems)"
        print(f"[MHc{args.mhc} shared-scores] {status}")
        for line in fails:
            print(f"    {line}")
        print(f"\nVerified MHc{args.mhc} shared-scores layout: "
              f"{len(fails)} problem(s) found.")
        return 1 if fails else 0

    if args.all:
        masspoints = [interpolation_config.masspoint_name(m, args.mhc)
                      for m in interpolation_config.study(args.mhc)["all"]]
    else:
        masspoints = [m.strip() for m in args.masspoints.split(",") if m.strip()]

    checked_shared = set()
    all_fails = []
    for mp in masspoints:
        fails = verify_masspoint(pnfs_samples, mp, checked_shared)
        status = "OK" if not fails else f"FAIL ({len(fails)} problems)"
        print(f"[{mp}] {status}")
        for line in fails:
            print(f"    {line}")
        all_fails.extend(fails)

    print(f"\nVerified {len(masspoints)} masspoint(s): "
          f"{len(all_fails)} problem(s) found.")
    return 1 if all_fails else 0


if __name__ == "__main__":
    sys.exit(main())
