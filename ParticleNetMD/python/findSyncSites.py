#!/usr/bin/env python
"""Name the GPU->CPU synchronization points of one GA training worker, by Python source line.

nsys counts ~44 cudaStreamSynchronize / cudaMemcpyAsync calls per training batch, but its
Python backtraces are not exported to SQLite. PyTorch can report them itself:
torch.cuda.set_sync_debug_mode("warn") raises a warning at every synchronizing operation,
from inside the Python call that caused it, so the stack names the exact source line.

This runs the real training code (trainWorker.train_epoch_disco / evaluate_disco) on a small
slice of the pilot data, in one process: one warm-up epoch, then one epoch of training and
validation with sync debugging on. Warnings are grouped by
  - the innermost frame in this repository (which line of our code triggered it), and
  - the innermost frame outside torch's own Python wrappers (e.g. torch_geometric / torch_cluster).

Limitation: only syncs that go through PyTorch's sync checks are seen (.item(), D2H copies,
nonzero, masked selection, ...). A C++/CUDA extension calling cudaMemcpy itself is invisible,
so compare the per-batch total with the nsys count.

Usage (GPU needed, ~2-3 min):
    findSyncSites.py --signal MHc130_MA90 --channel Combined --pilot
Output: profiling/syncs/sync_sites_<signal>_<channel>.csv and a printed table.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import argparse
import csv
import logging
import traceback
import warnings
from collections import Counter

import torch

from ProfileSetup import add_setup_arguments, build_training_setup, run_train_epoch, run_valid_epoch

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ParticleNetMD/
TORCH_DIR = os.path.dirname(torch.__file__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Find GPU->CPU sync sites in GA training, by source line")
    add_setup_arguments(parser)
    parser.add_argument("--output", default=None, help="CSV path (default: profiling/syncs/sync_sites_<sig>_<ch>.csv)")
    return parser.parse_args()


def short(frame):
    """file:line function, with long prefixes stripped."""
    path = frame.filename
    for prefix in (REPO_DIR + "/", os.path.dirname(TORCH_DIR) + "/"):
        if path.startswith(prefix):
            path = path[len(prefix):]
            break
    return f"{path}:{frame.lineno} {frame.name}"


def classify(stack):
    """(our innermost frame, innermost library frame outside torch's Python wrappers)."""
    ours = next((f for f in reversed(stack) if f.filename.startswith(REPO_DIR)
                 and not f.filename.endswith("findSyncSites.py")), None)
    # torch's own Python wrappers live in .../site-packages/torch/ (note the trailing slash:
    # .../site-packages/torch_geometric/ must NOT match)
    lib = next((f for f in reversed(stack)
                if not f.filename.startswith(TORCH_DIR + os.sep) and "warnings.py" not in f.filename), None)
    return (short(ours) if ours else "(none)"), (short(lib) if lib else "(none)")


def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s")
    setup = build_training_setup(args)
    n_train, n_valid = len(setup.train_loader), len(setup.valid_loader)

    phase = {"name": "warm-up"}

    def one_epoch():
        phase["name"] = "train"
        run_train_epoch(setup)
        phase["name"] = "valid"
        run_valid_epoch(setup)

    one_epoch()                                   # warm-up: allocator, cuBLAS heuristics, first kNN
    torch.cuda.synchronize()

    records = []

    def capture(message, category, filename, lineno, file=None, line=None):
        if "synchronizing CUDA operation" in str(message):
            records.append((phase["name"], traceback.extract_stack()[:-1]))

    previous_show = warnings.showwarning
    warnings.showwarning = capture
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = capture
        torch.cuda.set_sync_debug_mode("warn")
        try:
            one_epoch()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    warnings.showwarning = previous_show

    sites = Counter()
    for ph, stack in records:
        sites[(ph,) + classify(stack)] += 1
    batches = {"train": n_train, "valid": n_valid}
    per_phase = Counter(ph for ph, _ in records)

    print(f"\n{len(records)} synchronizing operations in one epoch: "
          + ", ".join(f"{ph} {per_phase[ph]} ({per_phase[ph] / batches[ph]:.1f} per batch, {batches[ph]} batches)"
                      for ph in ("train", "valid")))
    for ph in ("train", "valid"):
        print(f"\n[{ph}] {'per batch':>9s}  {'our code':50s}  innermost non-torch frame")
        for (p_, ours, lib), n in sites.most_common():
            if p_ == ph:
                print(f"        {n / batches[ph]:9.2f}  {ours[:50]:50s}  {lib if lib != ours else '(same)'}")

    out = args.output or os.path.join(REPO_DIR, "profiling", "syncs",
                                      f"sync_sites_{args.signal}_{args.channel}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase", "count", "per_batch", "our_frame", "innermost_non_torch_frame"])
        for (ph, ours, lib), n in sites.most_common():
            writer.writerow([ph, n, f"{n / batches[ph]:.3f}", ours, lib])
    print(f"\nWrote {out}")


if __name__ == "__main__":
    sys.exit(main())
