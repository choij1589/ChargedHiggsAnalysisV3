#!/usr/bin/env python
"""Single-process driver for Nsight Compute (ncu): one GA worker's training, in marked phases.

ncu replays each kernel many times and serializes launches, so it cannot profile the GA (16
processes, ~6 M launches). This trains one model the way a GA worker does (trainWorker.py code,
batch size from the GA config) on a small cached slice of the data, in two NVTX-marked phases:

    ncu_warmup   warm-up epochs: CUDA context, allocator growth, cuBLAS heuristics, first kNN
    ncu_measure  the epochs to profile

Profile only the measured phase with
    ncu --nvtx --nvtx-include "ncu_measure/" --kernel-name-base demangled --kernel-name regex:<k> ...
(scripts/ncuKernels.sh does this per kernel). The marker names contain no '/': ncu reads '/' as
the separator between nested ranges, so ranges like 'train/forward' cannot be used as filters.

Usage:
    profileKernels.py --signal MHc130_MA90 --channel Combined --pilot [--prepare-only]
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import argparse
import time

import torch
from torch.cuda import nvtx

from ProfileSetup import add_setup_arguments, build_training_setup, run_train_epoch, run_valid_epoch


def parse_arguments():
    parser = argparse.ArgumentParser(description="ncu driver: one GA worker's training in NVTX-marked phases")
    add_setup_arguments(parser)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--measure-epochs", type=int, default=1)
    parser.add_argument("--phase", choices=["train", "valid", "both"], default="both",
                        help="What runs inside each measured epoch (default: both)")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Only build the cached data slice, then exit (run this outside ncu)")
    args = parser.parse_args()
    if args.warmup_epochs < 1 or args.measure_epochs < 1:
        parser.error("--warmup-epochs and --measure-epochs must be >= 1")
    return args


def run_epoch(s, phase):
    if phase in ("train", "both"):
        run_train_epoch(s)
    if phase in ("valid", "both"):
        run_valid_epoch(s)


def main():
    args = parse_arguments()
    s = build_training_setup(args)
    if args.prepare_only:
        print("[profileKernels] data slice ready; exiting (--prepare-only)")
        return 0
    print(f"[profileKernels] {len(s.train_loader)} train + {len(s.valid_loader)} valid batches per epoch; "
          f"{args.warmup_epochs} warm-up + {args.measure_epochs} measured epoch(s), phase={args.phase}")

    nvtx.range_push("ncu_warmup")
    for _ in range(args.warmup_epochs):
        run_epoch(s, "both")
    torch.cuda.synchronize()
    nvtx.range_pop()

    start = time.time()
    nvtx.range_push("ncu_measure")
    for _ in range(args.measure_epochs):
        run_epoch(s, args.phase)
    torch.cuda.synchronize()
    nvtx.range_pop()
    print(f"[profileKernels] measured phase: {time.time() - start:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
