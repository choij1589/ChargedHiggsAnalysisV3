#!/usr/bin/env python
"""Single-process training setup shared by the profiling drivers (findSyncSites.py, profileKernels.py).

Builds what one GA worker builds (trainWorker.train_worker): a small train/valid slice of the
GA dataset as PyG Batches, DataLoaders, model, optimizer, scheduler and DisCo loss, and runs
epochs through the real trainWorker.train_epoch_disco / evaluate_disco.

The slice can be cached (torch.save of the two Batches), so repeated profiler runs, e.g. one ncu
invocation per kernel, do not reload the ~2 GB of pickled datasets each time.
"""

import os
import types

import torch

from GAConfig import load_ga_config
from DynamicDatasetLoader import DynamicDatasetLoader
from SharedDatasetManager import SharedDatasetManager
from SharedWorkerUtils import make_dataloader_from_batch
from MultiClassModels import create_multiclass_model
from WeightedLoss import DiScoWeightedLoss
from trainWorker import train_epoch_disco, evaluate_disco, create_optimizer, create_scheduler

PARTICLENETMD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_setup_arguments(parser):
    """Arguments shared by the profiling drivers."""
    parser.add_argument("--signal", required=True, help="Signal mass point, e.g. MHc130_MA90")
    parser.add_argument("--channel", default="Combined", help="Run1E2Mu, Run3Mu or Combined")
    parser.add_argument("--pilot", action="store_true", help="Use configs/GAConfig.pilot.json")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--events-per-class", type=int, default=2048,
                        help="Events per class and fold in the slice (default 2048 -> 16 batches of 512)")
    parser.add_argument("--nNodes", type=int, default=128)
    parser.add_argument("--optimizer", default="Adam", choices=["Adam", "RMSprop", "Adadelta"])
    parser.add_argument("--scheduler", default="ExponentialLR",
                        choices=["StepLR", "ExponentialLR", "CyclicLR", "ReduceLROnPlateau"])
    parser.add_argument("--initLR", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cache", default=None,
                        help="Cache file for the data slice (default: profiling/cache/slice_<sig>_<ch>_<n>[_pilot].pt; "
                             "'none' disables caching)")
    return parser


def _default_cache(args):
    tag = f"slice_{args.signal}_{args.channel}_{args.events_per_class}{'_pilot' if args.pilot else ''}.pt"
    return os.path.join(PARTICLENETMD_DIR, "profiling", "cache", tag)


def load_slice(args, config):
    """Return (train_batch, valid_batch): one training fold and the validation fold(s), capped per class."""
    cache = None if args.cache == "none" else (args.cache or _default_cache(args))
    if cache and os.path.exists(cache):
        saved = torch.load(cache, weights_only=False)
        print(f"[ProfileSetup] loaded data slice from {cache}")
        return saved["train"], saved["valid"]

    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")
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
    train_batch = manager._move_to_shared_memory(splits["train"], "train")
    valid_batch = manager._move_to_shared_memory(splits["valid"], "valid")
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save({"train": train_batch, "valid": valid_batch}, cache)
        print(f"[ProfileSetup] saved data slice to {cache}")
    return train_batch, valid_batch


def build_training_setup(args):
    """Everything one GA worker builds, for a single process on args.device."""
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is needed for the profiling drivers")
    config = load_ga_config(pilot=args.pilot)
    train_params = config.get_training_parameters()
    train_batch, valid_batch = load_slice(args, config)

    device = torch.device(args.device)
    num_classes = 1 + len(config.get_background_groups())
    model = create_multiclass_model(model_type=config.get_model_config()["default_model"],
                                    num_node_features=9, num_graph_features=8, num_classes=num_classes,
                                    num_hidden=args.nNodes, dropout_p=train_params["dropout_p"]).to(device)
    optimizer = create_optimizer(model, args.optimizer, args.initLR, args.weight_decay)
    return types.SimpleNamespace(
        model=model,
        optimizer=optimizer,
        scheduler=create_scheduler(optimizer, args.scheduler, args.optimizer, args.initLR),
        plateau=args.scheduler == "ReduceLROnPlateau",
        loss_fn=DiScoWeightedLoss(disco_lambda=config.get_disco_parameters()["disco_lambda"]),
        train_loader=make_dataloader_from_batch(train_batch, train_params["batch_size"], shuffle=True),
        valid_loader=make_dataloader_from_batch(valid_batch, train_params["batch_size"], shuffle=False),
        device=device,
        num_classes=num_classes,
        phi=train_params.get("augment_phi_rotation", True),
    )


def run_train_epoch(s):
    train_epoch_disco(s.model, s.train_loader, s.optimizer, s.scheduler, s.loss_fn, s.device,
                      s.plateau, s.num_classes, s.phi)


def run_valid_epoch(s):
    evaluate_disco(s.model, s.valid_loader, s.loss_fn, s.device, s.num_classes)
