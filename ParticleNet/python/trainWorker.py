#!/usr/bin/env python
"""
Worker function for GA optimization with shared memory datasets.

This module contains the worker function that trains individual models
using shared memory datasets, significantly reducing memory usage in GA optimization.
"""

import os
import json
import logging
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch

# Set PyTorch threads for worker processes
torch.set_num_threads(2)

from MultiClassModels import create_multiclass_model
from MLTools import EarlyStopper, SummaryWriter
from Preprocess import SharedBatchDataset


def create_optimizer(model, optimizer_name, initLR, weight_decay):
    """Create optimizer from configuration."""
    optimizers = {
        "RMSprop": lambda: torch.optim.RMSprop(model.parameters(), lr=initLR, momentum=0.9, weight_decay=weight_decay),
        "Adam": lambda: torch.optim.Adam(model.parameters(), lr=initLR, weight_decay=weight_decay),
        "Adadelta": lambda: torch.optim.Adadelta(model.parameters(), lr=initLR, weight_decay=weight_decay)
    }
    if optimizer_name not in optimizers:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    return optimizers[optimizer_name]()


def create_scheduler(optimizer, scheduler_name, optimizer_name, initLR):
    """Create LR scheduler from configuration."""
    schedulers = {
        "StepLR": lambda: torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95),
        "ExponentialLR": lambda: torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98),
        "CyclicLR": lambda: torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=initLR/5., max_lr=initLR*2, step_size_up=3, step_size_down=5,
            cycle_momentum=(optimizer_name == "RMSprop")
        ),
        "ReduceLROnPlateau": lambda: torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    }
    if scheduler_name not in schedulers:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")
    return schedulers[scheduler_name]()


def train_epoch(model, loader, optimizer, scheduler, device, use_plateau_scheduler):
    """Train for one epoch."""
    model.train()
    total_loss = 0.

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.graphInput, data.batch)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if use_plateau_scheduler:
        scheduler.step(total_loss)
    else:
        scheduler.step()

    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    """Evaluate model on dataset."""
    model.eval()
    loss = 0.
    correct = 0.

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.graphInput, data.batch)
            pred = out.argmax(dim=1)
            loss += float(F.cross_entropy(out, data.y, reduction='sum'))
            correct += int((pred == data.y).sum())

    return loss / len(loader.dataset), correct / len(loader.dataset)


def create_output_paths(worker_id, ga_config, args):
    """Create output directory paths for this worker."""
    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    output_config = ga_config.get_output_config()
    dataset_config = ga_config.get_dataset_config()
    signal_full = dataset_config['signal_prefix'] + args['signal']
    results_dir = output_config['results_dir']
    ga_subdir = output_config['ga_subdir_pattern'].format(iteration=args['iteration'])
    model_name = output_config['model_name_pattern'].format(idx=worker_id)

    base_dir = f"{WORKDIR}/ParticleNet/{results_dir}/{args['channel']}/multiclass/{signal_full}/{ga_subdir}"
    checkpoint_path = f"{base_dir}/{output_config['models_subdir']}/{model_name}.pt"
    json_path = f"{base_dir}/{output_config['json_subdir']}/{model_name}.json"

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    return checkpoint_path, json_path, model_name


def save_training_results(json_path, args, train_params, model_config, num_classes,
                         summary_writer, hyperparams, worker_id):
    """Save training results to JSON."""
    hyperparameters = {
        'signal': args['signal'],
        'channel': args['channel'],
        'iteration': args['iteration'],
        'model_idx': worker_id,
        'num_hidden': hyperparams['nNodes'],
        'optimizer': hyperparams['optimizer'],
        'initial_lr': hyperparams['initLR'],
        'weight_decay': hyperparams['weight_decay'],
        'scheduler': hyperparams['scheduler'],
        'pilot_mode': args['pilot'],
        'num_classes': num_classes,
        'model_type': model_config['default_model'],
        'num_node_features': 9,
        'num_graph_features': 4,
        'dropout_p': train_params['dropout_p'],
        'batch_size': train_params['batch_size'],
        'train_folds': train_params['train_folds'],
        'valid_folds': train_params['valid_folds']
    }

    best_metrics = summary_writer.get_best_metrics()
    output_data = {
        'hyperparameters': hyperparameters,
        'training_summary': {
            'best_epoch': best_metrics.get('best_epoch', -1),
            'best_train_loss': best_metrics.get('best_train_loss', -1),
            'best_valid_loss': best_metrics.get('best_valid_loss', -1),
            'best_train_acc': best_metrics.get('best_train_acc', -1),
            'best_valid_acc': best_metrics.get('best_valid_acc', -1),
            'total_epochs': len(summary_writer.metrics['epoch'])
        },
        'epoch_history': summary_writer.metrics
    }

    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)


def train_worker(worker_id, population_hyperparams, train_data_list, valid_data_list,
                args_dict, ga_config):
    """
    Worker function for parallel GA training with shared memory datasets.

    Args:
        worker_id: Unique worker identifier (model index in population)
        population_hyperparams: List of hyperparameter dictionaries for all models
        train_data_list: Shared memory training data list
        valid_data_list: Shared memory validation data list
        args_dict: Dictionary containing training arguments
        ga_config: GAConfig instance
    """
    # Setup logging for this worker
    logging.basicConfig(
        level=logging.DEBUG if args_dict.get('debug', False) else logging.INFO,
        format=f'[Worker {worker_id}] %(asctime)s - %(levelname)s - %(message)s'
    )

    # Get hyperparameters for this specific worker
    hyperparams = population_hyperparams[worker_id]

    logging.info(f"Starting training with hyperparameters:")
    logging.info(f"  nNodes={hyperparams['nNodes']}, optimizer={hyperparams['optimizer']}")
    logging.info(f"  initLR={hyperparams['initLR']}, weight_decay={hyperparams['weight_decay']}")
    logging.info(f"  scheduler={hyperparams['scheduler']}")

    # Get configurations
    train_params = ga_config.get_training_parameters()
    model_config = ga_config.get_model_config()
    bg_groups = ga_config.get_background_groups()
    num_classes = 1 + len(bg_groups)

    # Create datasets directly from shared Batch objects (NO UNBATCHING!)
    # This eliminates memory overhead from creating 491K individual Data objects
    logging.info(f"Creating datasets from shared Batch (no unbatching)...")
    train_dataset = SharedBatchDataset(train_data_list)
    valid_dataset = SharedBatchDataset(valid_data_list)

    logging.info(f"Train dataset: {len(train_dataset)} events (from shared memory)")
    logging.info(f"Valid dataset: {len(valid_dataset)} events (from shared memory)")

    # Create loaders with explicit collate function
    # collate_fn ensures mini-batches are properly batched into Batch objects
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_params['batch_size'],
        pin_memory=True,  # Faster CPU->GPU transfer
        shuffle=True,
        num_workers=0,  # Important: avoid nested multiprocessing
        collate_fn=Batch.from_data_list  # Explicit collation for mini-batches
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=train_params['batch_size'],
        pin_memory=True,
        shuffle=False,
        num_workers=0,  # Important: avoid nested multiprocessing
        collate_fn=Batch.from_data_list  # Explicit collation for mini-batches
    )

    # Setup device
    # Respect user-specified device - use only the requested GPU
    device = torch.device(args_dict['device'])
    logging.info(f"Using device {device} for worker {worker_id}")

    # Create model
    model = create_multiclass_model(
        model_type=model_config['default_model'],
        num_node_features=9,
        num_graph_features=4,
        num_classes=num_classes,
        num_hidden=hyperparams['nNodes'],
        dropout_p=train_params['dropout_p']
    ).to(device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        hyperparams['optimizer'],
        hyperparams['initLR'],
        hyperparams['weight_decay']
    )

    scheduler = create_scheduler(
        optimizer,
        hyperparams['scheduler'],
        hyperparams['optimizer'],
        hyperparams['initLR']
    )

    use_plateau_scheduler = (hyperparams['scheduler'] == "ReduceLROnPlateau")

    # Setup output paths
    checkpoint_path, json_path, model_name = create_output_paths(worker_id, ga_config, args_dict)

    # Setup early stopping and metrics
    early_stopper = EarlyStopper(
        patience=train_params['early_stopping_patience'],
        path=checkpoint_path
    )
    summary_writer = SummaryWriter(name=model_name)

    # Training loop
    logging.info(f"Starting training for up to {train_params['max_epochs']} epochs...")

    for epoch in range(train_params['max_epochs']):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, use_plateau_scheduler)

        # Evaluate
        valid_loss, valid_acc = evaluate(model, valid_loader, device)
        train_loss_eval, train_acc = evaluate(model, train_loader, device)

        # Log metrics
        summary_writer.log_metrics(epoch, train_loss_eval, valid_loss, train_acc, valid_acc)

        # Early stopping check
        early_stopper(valid_loss, model)

        if epoch % 10 == 0 or early_stopper.early_stop:
            logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
                        f"train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}")

        if early_stopper.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    final_valid_loss, final_valid_acc = evaluate(model, valid_loader, device)
    logging.info(f"Final validation: loss={final_valid_loss:.4f}, acc={final_valid_acc:.4f}")

    # Save results
    save_training_results(
        json_path, args_dict, train_params, model_config,
        num_classes, summary_writer, hyperparams, worker_id
    )

    logging.info(f"Worker {worker_id} completed successfully")
    logging.info(f"Results saved to {json_path}")

    # Clean up GPU memory if using CUDA
    if device.type == 'cuda':
        del model
        torch.cuda.empty_cache()

    return final_valid_acc  # Return validation accuracy for tracking
