#!/usr/bin/env python
"""
Lambda sweep launcher for ParticleNetMD.

Loads train/valid/test splits once into shared memory for a given signal,
then spawns one worker process per disco_lambda value (all sharing that
single in-memory dataset copy).

Memory comparison vs. old independent-process approach (8 lambdas):
  Old: 8 × 3 GB = 24 GB per signal
  New: 3 GB shared + 8 × 0.5 GB model state ≈ 7 GB per signal
"""

import argparse
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import torch.multiprocessing as mp

from trainMultiClass import Config
from DynamicDatasetLoader import DynamicDatasetLoader
from SharedDatasetManager import SharedDatasetManager
from SharedWorkerUtils import setup_spawn_method
from lambdaSweepWorker import lambda_sweep_worker


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Launch lambda sweep for ParticleNetMD (shared dataset)")
    parser.add_argument("--signal",   required=True, type=str,
                        help="Signal mass point (e.g., MHc130_MA90)")
    parser.add_argument("--channel",  required=True, type=str,
                        help="Channel (Run1E2Mu, Run3Mu, Combined)")
    parser.add_argument("--lambdas",  required=True, type=str,
                        help="Space-separated list of disco_lambda values, e.g. '0.0 0.005 0.01'")
    parser.add_argument("--config",   default=None, type=str,
                        help="Path to SglConfig JSON (default: configs/SglConfig.json)")
    parser.add_argument("--pilot",    action="store_true", default=False,
                        help="Pilot mode: reduced event caps, single train fold")
    return parser.parse_args()


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Parse lambda list
    lambdas = [float(x) for x in args.lambdas.split()]
    if not lambdas:
        raise ValueError("--lambdas must contain at least one value")

    logging.info("=" * 60)
    logging.info("Lambda sweep launcher")
    logging.info(f"  Signal:  {args.signal}")
    logging.info(f"  Channel: {args.channel}")
    logging.info(f"  Lambdas: {lambdas}")
    logging.info(f"  Pilot:   {args.pilot}")
    logging.info("=" * 60)

    # Load config to get fold lists and background groups
    # CLI --pilot is authoritative; override whatever the JSON says
    config = Config(args.signal, args.channel, args.config)
    config.args.pilot = args.pilot

    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    dataset_root = f"{WORKDIR}/ParticleNetMD/dataset"
    loader = DynamicDatasetLoader(dataset_root=dataset_root)

    background_groups_full = config.get_background_groups_full()

    # Pilot caps: match DataPipeline behaviour
    if config.args.pilot:
        train_folds = [config.args.train_folds[0]]
        max_events  = 8000   # per class per fold for train; 2000 for eval
    else:
        train_folds = config.args.train_folds
        max_events  = config.args.max_events_per_fold_per_class

    manager = SharedDatasetManager()

    # Load all three splits ONCE
    train_batch, valid_batch, test_batch = manager.prepare_shared_datasets(
        loader=loader,
        signal_sample=config.signal_full_name,
        background_groups=background_groups_full,
        channel=args.channel,
        train_folds=train_folds,
        valid_folds=config.args.valid_folds,
        test_folds=config.args.test_folds,
        pilot=config.args.pilot,
        max_events_per_fold=max_events,
        balance_weights=config.args.balance,
        random_state=42
    )

    memory_stats = manager.get_memory_usage()
    logging.info(f"Shared memory usage (train+valid): {memory_stats['total_gb']:.2f} GB")

    # Configure mp and spawn one worker per lambda
    setup_spawn_method()

    logging.info(f"Spawning {len(lambdas)} worker processes...")
    try:
        mp.spawn(
            lambda_sweep_worker,
            args=(lambdas, train_batch, valid_batch, test_batch,
                  args.signal, args.channel, args.config, args.pilot),
            nprocs=len(lambdas),
            join=True
        )
    except Exception as e:
        logging.error(f"Lambda sweep failed: {e}")
        raise

    logging.info("All lambda sweep workers completed successfully")


if __name__ == "__main__":
    main()
