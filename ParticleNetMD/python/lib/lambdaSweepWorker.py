#!/usr/bin/env python
"""
Worker function for lambda sweep training with shared datasets in ParticleNetMD.

Each worker handles one value of disco_lambda, sharing the dataset loaded by
launchLambdaSweep.py, so dataset I/O happens only once per signal.
"""

import logging

import torch
from torch_geometric.data import Batch

from trainMultiClass import Config
from DataPipeline import DataPipeline
from TrainingOrchestrator import create_training_orchestrator
from ResultPersistence import create_result_persistence
from SharedWorkerUtils import make_dataloader_from_batch

# Reduce intra-op parallelism so workers don't fight for CPU threads
torch.set_num_threads(1)


def lambda_sweep_worker(rank: int,
                        lambdas,
                        train_batch: Batch,
                        valid_batch: Batch,
                        test_batch: Batch,
                        signal: str,
                        channel: str,
                        config_path: str,
                        pilot: bool):
    """
    Train one model for a single disco_lambda value.

    Args:
        rank: Worker index — selects lambdas[rank] as disco_lambda
        lambdas: Sequence of lambda values (one per worker)
        train_batch: Shared-memory Batch for training split
        valid_batch: Shared-memory Batch for validation split
        test_batch:  Shared-memory Batch for test split
        signal: Signal mass point (e.g., "MHc130_MA90")
        channel: Channel string (Run1E2Mu, Run3Mu, Combined)
        config_path: Path to SglConfig JSON (or None for default)
        pilot: Whether to run in pilot mode
    """
    disco_lambda = lambdas[rank]

    logging.basicConfig(
        level=logging.INFO,
        format=f'[lambda={disco_lambda}] %(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Worker {rank}: disco_lambda={disco_lambda}")

    # Build config and apply overrides
    config = Config(signal, channel, config_path)
    config.args.disco_lambda = disco_lambda
    config.args.pilot = pilot

    # Build DataLoaders from shared batches — no disk I/O
    train_loader = make_dataloader_from_batch(train_batch, config.args.batch_size, shuffle=True)
    valid_loader = make_dataloader_from_batch(valid_batch, config.args.batch_size, shuffle=False)
    test_loader  = make_dataloader_from_batch(test_batch,  config.args.batch_size, shuffle=False)

    # Inject into DataPipeline, bypassing file loading
    pipeline = DataPipeline(config)
    pipeline.inject_loaders(train_loader, valid_loader, test_loader)

    # Determine output paths (identical scheme as trainMultiClass.py)
    model_name   = config.get_model_name()
    output_paths = config.get_output_paths(model_name)
    output_path, checkpoint_path, summary_path, tree_path = output_paths

    logging.info(f"Model name: {model_name}")
    logging.info(f"Output path: {output_path}")

    # Setup persistence and orchestrator
    persistence  = create_result_persistence(config)
    persistence.create_output_directories(output_paths)

    orchestrator = create_training_orchestrator(config, pipeline)
    orchestrator.setup_training_infrastructure(model_name, checkpoint_path)

    # Train
    training_results = orchestrator.train()
    test_results     = orchestrator.evaluate_final_performance()
    training_results.update(test_results)

    # Save outputs (identical to trainMultiClass.py steps 9–12)
    orchestrator.save_training_summary(summary_path)

    model  = orchestrator.get_model()
    device = orchestrator.device
    persistence.save_predictions_to_root(model, pipeline, device, tree_path)

    persistence.save_performance_summary(training_results, model_name, output_path)
    persistence.save_model_info(model, model_name, output_path)
    persistence.save_ga_compatible_json(training_results, model_name, output_path)

    logging.info(f"Training complete  lambda={disco_lambda}  -> {tree_path}")

    # Free GPU memory before next worker starts
    if device.type == 'cuda':
        del model
        torch.cuda.empty_cache()
