#!/usr/bin/env python
"""Multi-class training script for Genetic Algorithm optimization."""

import os
import argparse
import logging
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.utils import shuffle

from GAConfig import load_ga_config
from MultiClassModels import create_multiclass_model
from MLTools import EarlyStopper, SummaryWriter
from DynamicDatasetLoader import DynamicDatasetLoader
from Preprocess import GraphDataset


def parse_arguments():
    """Parse command line arguments for GA training."""
    parser = argparse.ArgumentParser(description="Multi-class training for GA optimization")
    parser.add_argument("--signal", required=True, type=str, help="Signal sample (e.g., MHc130_MA100)")
    parser.add_argument("--channel", required=True, type=str, help="Channel (Run1E2Mu, Run3Mu, Combined)")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cpu or cuda:X)")
    parser.add_argument("--iter", required=True, type=int, help="GA iteration number")
    parser.add_argument("--idx", required=True, type=int, help="Model index in population")
    parser.add_argument("--nNodes", required=True, type=int, help="Number of hidden nodes")
    parser.add_argument("--optimizer", required=True, type=str, help="Optimizer (Adam, RMSprop, Adadelta)")
    parser.add_argument("--initLR", required=True, type=float, help="Initial learning rate")
    parser.add_argument("--weight_decay", required=True, type=float, help="Weight decay")
    parser.add_argument("--scheduler", required=True, type=str, help="LR scheduler")
    parser.add_argument("--pilot", action="store_true", default=False, help="Use pilot datasets")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")
    return parser.parse_args()


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


def load_multiclass_datasets(loader, signal_full, background_groups_full, channel, fold_list, pilot):
    """Load and combine data from multiple folds."""
    all_data = []

    # Load signal data
    for fold in fold_list:
        signal_data = loader.load_sample_data(signal_full, "signal", channel, fold, pilot)
        for data in signal_data:
            data.y = torch.tensor(0, dtype=torch.long)  # Signal = class 0
        all_data.extend(signal_data)
        logging.info(f"Loaded {len(signal_data)} signal events from fold {fold}")

    # Load background groups
    for group_idx, (group_name, sample_list) in enumerate(background_groups_full.items()):
        group_label = group_idx + 1  # Background groups: 1, 2, 3, ...

        for fold in fold_list:
            group_fold_data = []
            for sample_name in sample_list:
                sample_data = loader.load_sample_data(sample_name, "background", channel, fold, pilot)
                group_fold_data.extend(sample_data)

            # Assign group labels
            for data in group_fold_data:
                data.y = torch.tensor(group_label, dtype=torch.long)

            all_data.extend(group_fold_data)
            logging.info(f"Loaded {len(group_fold_data)} events from {group_name} (label {group_label}) fold {fold}")

    return shuffle(all_data, random_state=42)


def setup_training_environment(args, ga_config):
    """Setup training environment and load datasets."""
    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    # Get configurations
    train_params = ga_config.get_training_parameters()
    bg_groups = ga_config.get_background_groups()
    dataset_config = ga_config.get_dataset_config()

    # Construct full sample names
    signal_full = dataset_config['signal_prefix'] + args.signal
    background_full_names = [dataset_config['background_prefix'] + bg for bg in sum(bg_groups.values(), [])]
    background_groups_full = {
        group_name: [dataset_config['background_prefix'] + sample for sample in samples]
        for group_name, samples in bg_groups.items()
    }

    logging.info(f"Signal: {signal_full}")
    logging.info(f"Model {args.idx} - Iteration {args.iter}")
    logging.info(f"Hyperparameters: nNodes={args.nNodes}, opt={args.optimizer}, "
                f"lr={args.initLR}, wd={args.weight_decay}, sched={args.scheduler}")

    # Load datasets
    dataset_root = f"{WORKDIR}/ParticleNet/dataset"
    loader = DynamicDatasetLoader(dataset_root=dataset_root, separate_bjets=dataset_config['use_bjets'])

    # Load and combine data
    train_data = load_multiclass_datasets(
        loader, signal_full, background_groups_full, args.channel, train_params['train_folds'], args.pilot
    )
    valid_data = load_multiclass_datasets(
        loader, signal_full, background_groups_full, args.channel, train_params['valid_folds'], args.pilot
    )

    # Create datasets and loaders
    train_dataset = GraphDataset(train_data)
    valid_dataset = GraphDataset(valid_data)
    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=train_params['batch_size'], pin_memory=True, shuffle=False)

    logging.info(f"Train dataset: {len(train_dataset)} events")
    logging.info(f"Valid dataset: {len(valid_dataset)} events")

    return train_loader, valid_loader, train_params, bg_groups, dataset_config


def create_output_paths(WORKDIR, ga_config, args):
    """Create output directory paths."""
    output_config = ga_config.get_output_config()
    dataset_config = ga_config.get_dataset_config()
    signal_full = dataset_config['signal_prefix'] + args.signal
    results_dir = output_config['results_dir']
    ga_subdir = output_config['ga_subdir_pattern'].format(iteration=args.iter)
    model_name = output_config['model_name_pattern'].format(idx=args.idx)

    base_dir = f"{WORKDIR}/ParticleNet/{results_dir}/{args.channel}/multiclass/{signal_full}/{ga_subdir}"
    checkpoint_path = f"{base_dir}/{output_config['models_subdir']}/{model_name}.pt"
    json_path = f"{base_dir}/{output_config['json_subdir']}/{model_name}.json"

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    return checkpoint_path, json_path, model_name


def save_training_results(json_path, args, train_params, model_config, num_classes, summary_writer):
    """Save training results to JSON."""
    import json

    hyperparameters = {
        'signal': args.signal, 'channel': args.channel, 'iteration': args.iter, 'model_idx': args.idx,
        'num_hidden': args.nNodes, 'optimizer': args.optimizer, 'initial_lr': args.initLR,
        'weight_decay': args.weight_decay, 'scheduler': args.scheduler, 'pilot_mode': args.pilot,
        'num_classes': num_classes, 'model_type': model_config['default_model'],
        'num_node_features': 9, 'num_graph_features': 4, 'dropout_p': train_params['dropout_p'],
        'batch_size': train_params['batch_size'], 'train_folds': train_params['train_folds'],
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
    logging.info(f"Training complete. Results saved to {json_path}")


def main():
    """Main training function for GA optimization."""
    args = parse_arguments()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.pilot:
        logging.info("=" * 60)
        logging.info("PILOT MODE ENABLED - Using pilot datasets")
        logging.info("=" * 60)

    # Load configuration and setup environment
    ga_config = load_ga_config()
    train_loader, valid_loader, train_params, bg_groups, dataset_config = setup_training_environment(args, ga_config)

    # Setup device
    if "cuda" in args.device:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

    # Create model
    model_config = ga_config.get_model_config()
    num_classes = 1 + len(bg_groups)
    model = create_multiclass_model(
        model_type=model_config['default_model'],
        num_node_features=9,
        num_graph_features=4,
        num_classes=num_classes,
        num_hidden=args.nNodes,
        dropout_p=train_params['dropout_p']
    ).to(args.device)

    logging.info(f"Model: {model_config['default_model']} with {num_classes} classes")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, args.optimizer, args.initLR, args.weight_decay)
    scheduler = create_scheduler(optimizer, args.scheduler, args.optimizer, args.initLR)
    use_plateau_scheduler = (args.scheduler == "ReduceLROnPlateau")

    # Setup output paths
    WORKDIR = os.environ.get("WORKDIR")
    checkpoint_path, json_path, model_name = create_output_paths(WORKDIR, ga_config, args)

    # Setup early stopper and summary writer
    early_stopper = EarlyStopper(patience=train_params['early_stopping_patience'], path=checkpoint_path)
    summary_writer = SummaryWriter(name=model_name)

    # Training loop
    logging.info("Starting training...")
    for epoch in range(train_params['max_epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, args.device, use_plateau_scheduler)
        valid_loss, valid_acc = evaluate(model, valid_loader, args.device)
        train_loss_eval, train_acc = evaluate(model, train_loader, args.device)

        summary_writer.log_metrics(epoch, train_loss_eval, valid_loss, train_acc, valid_acc)

        early_stopper(valid_loss, model)
        if early_stopper.early_stop:
            logging.info(f"Early stopping at epoch {epoch}")
            break

    # Save results
    save_training_results(json_path, args, train_params, model_config, num_classes, summary_writer)


if __name__ == "__main__":
    main()
