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
from collections import Counter, defaultdict

import torch

from GAConfig import load_ga_config
from DynamicDatasetLoader import DynamicDatasetLoader
from SharedDatasetManager import SharedDatasetManager
from SharedWorkerUtils import make_dataloader_from_batch
from MultiClassModels import create_multiclass_model
from WeightedLoss import DiScoWeightedLoss
from trainWorker import train_epoch_disco, evaluate_disco, create_optimizer, create_scheduler

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # ParticleNetMD/
TORCH_DIR = os.path.dirname(torch.__file__)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Find GPU->CPU sync sites in GA training, by source line")
    parser.add_argument("--signal", required=True, help="Signal mass point, e.g. MHc130_MA90")
    parser.add_argument("--channel", default="Combined", help="Run1E2Mu, Run3Mu or Combined")
    parser.add_argument("--pilot", action="store_true", help="Use configs/GAConfig.pilot.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--events-per-class", type=int, default=2048,
                        help="Events per class and fold for the slice (default 2048 -> 16 batches of 512)")
    parser.add_argument("--nNodes", type=int, default=128)
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "RMSprop", "Adadelta"])
    parser.add_argument("--scheduler", default="ExponentialLR",
                        choices=["StepLR", "ExponentialLR", "CyclicLR", "ReduceLROnPlateau"])
    parser.add_argument("--initLR", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    if not torch.cuda.is_available():
        raise RuntimeError("findSyncSites.py needs a GPU")
    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    config = load_ga_config(pilot=args.pilot)
    dataset_config, train_params = config.get_dataset_config(), config.get_training_parameters()
    groups = {g: [dataset_config["background_prefix"] + s for s in samples]
              for g, samples in config.get_background_groups().items()}
    loader = DynamicDatasetLoader(dataset_root=f"{WORKDIR}/ParticleNetMD/dataset")
    splits = loader.load_multiclass_splits(
        dataset_config["signal_prefix"] + args.signal, groups, args.channel,
        {"train": ([train_params["train_folds"][0]], args.events_per_class),
         "valid": (train_params["valid_folds"], args.events_per_class)},
        balance_weights=train_params.get("balance_weights", True))
    manager = SharedDatasetManager()
    train_loader = make_dataloader_from_batch(manager._move_to_shared_memory(splits["train"], "train"),
                                              train_params["batch_size"], shuffle=True)
    valid_loader = make_dataloader_from_batch(manager._move_to_shared_memory(splits["valid"], "valid"),
                                              train_params["batch_size"], shuffle=False)
    n_train, n_valid = len(train_loader), len(valid_loader)

    device = torch.device(args.device)
    num_classes = 1 + len(groups)
    model = create_multiclass_model(model_type=config.get_model_config()["default_model"],
                                    num_node_features=9, num_graph_features=8, num_classes=num_classes,
                                    num_hidden=args.nNodes, dropout_p=train_params["dropout_p"]).to(device)
    optimizer = create_optimizer(model, args.optimizer, args.initLR, args.weight_decay)
    scheduler = create_scheduler(optimizer, args.scheduler, args.optimizer, args.initLR)
    plateau = args.scheduler == "ReduceLROnPlateau"
    loss_fn = DiScoWeightedLoss(disco_lambda=config.get_disco_parameters()["disco_lambda"])
    phi = train_params.get("augment_phi_rotation", True)

    phase = {"name": "warm-up"}

    def one_epoch():
        phase["name"] = "train"
        train_epoch_disco(model, train_loader, optimizer, scheduler, loss_fn, device, plateau, num_classes, phi)
        phase["name"] = "valid"
        evaluate_disco(model, valid_loader, loss_fn, device, num_classes)

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
