#!/usr/bin/env python
"""
Worker function for GA optimization with shared memory datasets in ParticleNetMD.

This module contains the worker function that trains individual models
using shared memory datasets with DisCo loss for mass decorrelation.

Key features for ParticleNetMD:
- DisCo loss with mass tensors (mass1, mass2)
- Group-balanced accuracy metric
- Decomposed loss tracking (CE + DisCo)
"""

import os
import json
import logging
import torch

# Set PyTorch threads for worker processes
torch.set_num_threads(1)

from MultiClassModels import create_multiclass_model
from MLTools import EarlyStopper, SummaryWriter
from WeightedLoss import DiScoWeightedLoss
from TrainingUtilities import calculate_group_balanced_accuracy, apply_phi_rotation
from SharedWorkerUtils import make_dataloader_from_batch


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


def train_epoch_disco(model, loader, optimizer, scheduler, loss_fn, device, use_plateau_scheduler, num_classes,
                      augment_phi_rotation=True):
    """
    Train for one epoch with DisCo loss.

    Args:
        model: PyTorch model
        loader: DataLoader
        optimizer: Optimizer
        scheduler: LR scheduler
        loss_fn: DiScoWeightedLoss instance
        device: torch.device
        use_plateau_scheduler: Whether using ReduceLROnPlateau
        num_classes: Number of classes for accuracy calculation
        augment_phi_rotation: Whether to apply random phi rotation augmentation

    Returns:
        Tuple of (avg_loss, group_balanced_accuracy, decomposed_losses_dict)
    """
    model.train()

    loss_accumulator = []
    ce_accumulator = []
    disco_accumulator = []
    all_predictions = []
    all_labels = []
    all_weights = []

    for batch in loader:
        batch = batch.to(device)

        # Apply phi rotation augmentation (training only)
        if augment_phi_rotation:
            batch.x = apply_phi_rotation(batch.x, batch.batch, device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)

        # Get tensors for DisCo loss
        weights = batch.weight
        mass1 = batch.mass1.squeeze()
        mass2 = batch.mass2.squeeze()

        # Compute DisCo loss
        loss = loss_fn(logits, batch.y, weights, mass1, mass2)

        loss.backward()
        optimizer.step()

        # Track losses
        loss_accumulator.append(loss.detach())
        ce_accumulator.append(loss_fn.last_ce_loss)
        disco_accumulator.append(loss_fn.last_disco_term)

        # Collect for accuracy
        pred = logits.argmax(dim=1)
        all_predictions.append(pred.detach())
        all_labels.append(batch.y)
        all_weights.append(weights)

    # Scheduler step
    total_loss = torch.stack(loss_accumulator).sum().item()
    if use_plateau_scheduler:
        scheduler.step(total_loss)
    else:
        scheduler.step()

    # Calculate group-balanced accuracy
    all_predictions = torch.cat(all_predictions).cpu()
    all_labels = torch.cat(all_labels).cpu()
    all_weights = torch.cat(all_weights).cpu()

    accuracy = calculate_group_balanced_accuracy(
        all_predictions, all_labels, all_weights,
        use_groups=True, num_classes=num_classes
    )

    # Average losses
    avg_loss = total_loss / len(loader)
    avg_ce = sum(ce_accumulator) / len(ce_accumulator) if ce_accumulator else 0.0
    avg_disco = sum(disco_accumulator) / len(disco_accumulator) if disco_accumulator else 0.0

    return avg_loss, accuracy, {'ce_loss': avg_ce, 'disco_term': avg_disco}


def evaluate_disco(model, loader, loss_fn, device, num_classes):
    """
    Evaluate model with DisCo loss and group-balanced accuracy.

    Args:
        model: PyTorch model
        loader: DataLoader
        loss_fn: DiScoWeightedLoss instance
        device: torch.device
        num_classes: Number of classes

    Returns:
        Tuple of (avg_loss, group_balanced_accuracy, decomposed_losses_dict)
    """
    model.eval()

    loss_accumulator = []
    ce_accumulator = []
    disco_accumulator = []
    all_predictions = []
    all_labels = []
    all_weights = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)

            # Get tensors for DisCo loss
            weights = batch.weight
            mass1 = batch.mass1.squeeze()
            mass2 = batch.mass2.squeeze()

            # Compute DisCo loss
            loss = loss_fn(logits, batch.y, weights, mass1, mass2)

            # Track losses (detach to free GPU memory)
            loss_accumulator.append(loss.detach())
            ce_accumulator.append(loss_fn.last_ce_loss)
            disco_accumulator.append(loss_fn.last_disco_term)

            # Collect for accuracy
            pred = logits.argmax(dim=1)
            all_predictions.append(pred)
            all_labels.append(batch.y)
            all_weights.append(weights)

    # Calculate group-balanced accuracy
    all_predictions = torch.cat(all_predictions).cpu()
    all_labels = torch.cat(all_labels).cpu()
    all_weights = torch.cat(all_weights).cpu()

    accuracy = calculate_group_balanced_accuracy(
        all_predictions, all_labels, all_weights,
        use_groups=True, num_classes=num_classes
    )

    # Average losses
    total_loss = torch.stack(loss_accumulator).sum().item()
    avg_loss = total_loss / len(loader)
    avg_ce = sum(ce_accumulator) / len(ce_accumulator) if ce_accumulator else 0.0
    avg_disco = sum(disco_accumulator) / len(disco_accumulator) if disco_accumulator else 0.0

    return avg_loss, accuracy, {'ce_loss': avg_ce, 'disco_term': avg_disco}


def create_output_paths(worker_id, ga_config, args):
    """Create output directory paths for this worker."""
    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    output_config = ga_config.get_output_config()
    results_dir = output_config['results_dir']
    ga_subdir = output_config['ga_subdir_pattern'].format(iteration=args['iteration'])
    model_name = output_config['model_name_pattern'].format(idx=worker_id)

    # ParticleNetMD-specific path: short signal name, fold/pilot at leaf level
    test_folds = args.get('test_folds', [4])
    fold_dir = "pilot" if args.get('pilot') else f"fold-{test_folds[0]}"
    base_dir = f"{WORKDIR}/ParticleNetMD/{results_dir}/{args['channel']}/{args['signal']}/{fold_dir}/{ga_subdir}"
    checkpoint_path = f"{base_dir}/{output_config['models_subdir']}/{model_name}.pt"
    json_path = f"{base_dir}/{output_config['json_subdir']}/{model_name}.json"

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    return checkpoint_path, json_path, model_name


def save_training_results(json_path, args, train_params, model_config, disco_params, num_classes,
                          summary_writer, hyperparams, worker_id):
    """Save training results to JSON with decomposed losses."""
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
        'num_graph_features': 8,
        'dropout_p': train_params['dropout_p'],
        'batch_size': train_params['batch_size'],
        'train_folds': train_params['train_folds'],
        'valid_folds': train_params['valid_folds'],
        # ParticleNetMD-specific
        'loss_type': train_params['loss_type'],
        'disco_lambda': disco_params['disco_lambda']
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
    Worker function for parallel GA training with DisCo loss.

    Args:
        worker_id: Unique worker identifier (model index in population)
        population_hyperparams: List of hyperparameter dictionaries for all models
        train_data_list: Shared memory training Batch object
        valid_data_list: Shared memory validation Batch object
        args_dict: Dictionary containing training arguments (includes disco_lambda)
        ga_config: GAConfigLoader instance
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
    disco_params = ga_config.get_disco_parameters()
    bg_groups = ga_config.get_background_groups()
    num_classes = 1 + len(bg_groups)

    logging.info(f"DisCo lambda: {disco_params['disco_lambda']} (fixed, not optimized)")
    logging.info(f"Number of classes: {num_classes}")

    # Create DataLoaders from shared Batch objects (NO UNBATCHING!)
    logging.info(f"Creating DataLoaders from shared Batch (no unbatching)...")
    train_loader = make_dataloader_from_batch(train_data_list, train_params['batch_size'], shuffle=True)
    valid_loader = make_dataloader_from_batch(valid_data_list, train_params['batch_size'], shuffle=False)

    logging.info(f"Train dataset: {len(train_loader.dataset)} events (from shared memory)")
    logging.info(f"Valid dataset: {len(valid_loader.dataset)} events (from shared memory)")

    # Setup device
    device = torch.device(args_dict['device'])
    logging.info(f"Using device {device} for worker {worker_id}")

    # Create model
    model = create_multiclass_model(
        model_type=model_config['default_model'],
        num_node_features=9,
        num_graph_features=8,
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

    # Create DisCo loss function
    loss_fn = DiScoWeightedLoss(disco_lambda=disco_params['disco_lambda'])

    # Setup output paths
    checkpoint_path, json_path, model_name = create_output_paths(worker_id, ga_config, args_dict)

    # Setup early stopping and metrics
    early_stopper = EarlyStopper(
        patience=train_params['early_stopping_patience'],
        path=checkpoint_path
    )
    summary_writer = SummaryWriter(name=model_name)

    # Add decomposed loss tracking to metrics
    summary_writer.metrics['train_ce_loss'] = []
    summary_writer.metrics['valid_ce_loss'] = []
    summary_writer.metrics['train_disco_term'] = []
    summary_writer.metrics['valid_disco_term'] = []

    # Training loop
    max_epochs = 10 if args_dict.get('pilot') else train_params['max_epochs']
    if args_dict.get('pilot'):
        logging.info(f"PILOT MODE: max_epochs capped to {max_epochs}")
    logging.info(f"Starting training for up to {max_epochs} epochs...")

    for epoch in range(max_epochs):
        # Train
        train_loss, train_acc, train_decomposed = train_epoch_disco(
            model, train_loader, optimizer, scheduler, loss_fn,
            device, use_plateau_scheduler, num_classes,
            args_dict.get('augment_phi_rotation', True)
        )

        # Evaluate
        valid_loss, valid_acc, valid_decomposed = evaluate_disco(
            model, valid_loader, loss_fn, device, num_classes
        )

        # Log metrics
        summary_writer.log_metrics(epoch, train_loss, valid_loss, train_acc, valid_acc)

        # Log decomposed losses
        summary_writer.metrics['train_ce_loss'].append(train_decomposed['ce_loss'])
        summary_writer.metrics['valid_ce_loss'].append(valid_decomposed['ce_loss'])
        summary_writer.metrics['train_disco_term'].append(train_decomposed['disco_term'])
        summary_writer.metrics['valid_disco_term'].append(valid_decomposed['disco_term'])

        # Early stopping check
        early_stopper(valid_loss, model)

        if epoch % 10 == 0 or early_stopper.early_stop:
            logging.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, "
                         f"train_acc={train_acc:.4f}, valid_acc={valid_acc:.4f}")
            logging.info(f"  CE: train={train_decomposed['ce_loss']:.4f}, valid={valid_decomposed['ce_loss']:.4f}")
            logging.info(f"  DisCo: train={train_decomposed['disco_term']:.4f}, valid={valid_decomposed['disco_term']:.4f}")

        if early_stopper.early_stop:
            logging.info(f"Early stopping triggered at epoch {epoch}")
            break

    # Load best model
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Final evaluation
    final_valid_loss, final_valid_acc, final_decomposed = evaluate_disco(
        model, valid_loader, loss_fn, device, num_classes
    )
    logging.info(f"Final validation: loss={final_valid_loss:.4f}, acc={final_valid_acc:.4f}")
    logging.info(f"  CE={final_decomposed['ce_loss']:.4f}, DisCo={final_decomposed['disco_term']:.4f}")

    # Save results
    save_training_results(
        json_path, args_dict, train_params, model_config, disco_params,
        num_classes, summary_writer, hyperparams, worker_id
    )

    logging.info(f"Worker {worker_id} completed successfully")
    logging.info(f"Results saved to {json_path}")

    # Clean up GPU memory
    if device.type == 'cuda':
        del model
        torch.cuda.empty_cache()

    return final_valid_acc
