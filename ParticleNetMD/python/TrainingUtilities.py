#!/usr/bin/env python
"""
Core training utilities for Mass-Decorrelated ParticleNet training.

Contains training functions, evaluation metrics, performance monitoring,
and group-balanced accuracy calculations for physics-aware training.

Modified from ParticleNet to support DisCo loss with mass tensors.
"""

import time
import logging
import psutil
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn.functional as F


def extract_weights_from_batch(batch):
    """Extract weights from a batch for loss function."""
    if hasattr(batch, 'weight'):
        return batch.weight
    else:
        # Fallback: uniform weights
        return torch.ones(batch.y.size(0), device=batch.y.device)


def calculate_group_balanced_accuracy(predictions, labels, weights, use_groups, num_classes):
    """
    Calculate group-balanced accuracy for multi-class classification.

    For grouped backgrounds:
    - Within each group, accuracy is weighted by sample weights
    - Between groups, each group contributes equally (1/num_classes)

    Args:
        predictions: predicted class labels (tensor)
        labels: true class labels (tensor)
        weights: sample weights (tensor)
        use_groups: whether using grouped backgrounds
        num_classes: total number of classes

    Returns:
        group_balanced_accuracy: float
    """
    if not use_groups:
        # Fall back to standard weighted accuracy for individual backgrounds
        correct_mask = (predictions == labels).float()
        weighted_correct = (correct_mask * weights).sum()
        total_weights = weights.sum()
        return (weighted_correct / total_weights).item() if total_weights > 0 else 0.0

    # Group-balanced accuracy calculation
    class_accuracies = []

    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        if class_mask.sum() == 0:
            continue  # Skip classes with no samples

        class_predictions = predictions[class_mask]
        class_labels = labels[class_mask]
        class_weights = weights[class_mask]

        # Within-class weighted accuracy
        correct_mask = (class_predictions == class_labels).float()
        weighted_correct = (correct_mask * class_weights).sum()
        total_class_weight = class_weights.sum()

        if total_class_weight > 0:
            class_accuracy = (weighted_correct / total_class_weight).item()
            class_accuracies.append(class_accuracy)

    # Return mean accuracy across all classes (equal weight per group)
    return sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0


def log_detailed_class_accuracies(predictions, labels, weights, use_groups, num_classes,
                                background_groups=None, epoch=None, prefix=""):
    """
    Log detailed per-class accuracy information for debugging and monitoring.

    Args:
        predictions: predicted class labels (tensor)
        labels: true class labels (tensor)
        weights: sample weights (tensor)
        use_groups: whether using grouped backgrounds
        num_classes: total number of classes
        background_groups: dict of background groups (for naming)
        epoch: current epoch number (optional)
        prefix: prefix for log messages (e.g., "TRAIN" or "VALID")
    """
    if not use_groups:
        return  # Skip detailed logging for individual backgrounds

    class_names = ["signal"]
    if background_groups:
        class_names.extend(list(background_groups.keys()))
    else:
        class_names.extend([f"bg_{i}" for i in range(1, num_classes)])

    epoch_str = f"[EPOCH {epoch}] " if epoch is not None else ""
    logging.info(f"{epoch_str}{prefix} Per-Class Accuracies (Group-Balanced):")

    class_accuracies = []
    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        class_count = class_mask.sum().item()

        if class_count == 0:
            logging.info(f"  {class_names[class_idx]:>12}: No samples")
            continue

        class_predictions = predictions[class_mask]
        class_labels = labels[class_mask]
        class_weights = weights[class_mask]

        # Within-class weighted accuracy
        correct_mask = (class_predictions == class_labels).float()
        weighted_correct = (correct_mask * class_weights).sum()
        total_class_weight = class_weights.sum()

        if total_class_weight > 0:
            class_accuracy = (weighted_correct / total_class_weight).item()
            class_accuracies.append(class_accuracy)
            logging.info(f"  {class_names[class_idx]:>12}: {class_accuracy*100:6.2f}% "
                        f"({class_count:6d} samples, weight: {total_class_weight.item():8.2f})")
        else:
            logging.info(f"  {class_names[class_idx]:>12}: No weight")

    overall_accuracy = sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0
    logging.info(f"  {'OVERALL':>12}: {overall_accuracy*100:6.2f}% (group-balanced mean)")


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device,
               use_groups=False, num_classes=4, scheduler_type="StepLR",
               loss_requires_mass=False):
    """
    Train model for one epoch.

    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        loss_fn: Loss function
        device: Training device (cpu/cuda)
        use_groups: Whether using grouped backgrounds
        num_classes: Number of classes
        scheduler_type: Type of scheduler for proper step handling
        loss_requires_mass: Whether loss function needs mass tensors (DisCo)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()

    # Accumulate losses on GPU to avoid per-batch synchronization
    loss_accumulator = []

    # Accumulate decomposed losses for DisCo tracking
    ce_loss_accumulator = []
    disco_term_accumulator = []
    track_decomposed = loss_requires_mass and hasattr(loss_fn, 'get_decomposed_losses')

    # Collect all predictions, labels, and weights for group-balanced accuracy
    # Keep on GPU during training to avoid unnecessary CPU transfers
    all_predictions = []
    all_labels = []
    all_weights = []

    for batch in train_loader:
        batch = batch.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)

        # Extract weights for loss function
        weights = extract_weights_from_batch(batch)

        # Compute weighted loss (with or without mass for DisCo)
        if loss_requires_mass:
            mass1 = batch.mass1.squeeze()
            mass2 = batch.mass2.squeeze()
            loss = loss_fn(logits, batch.y, weights, mass1, mass2)
        else:
            loss = loss_fn(logits, batch.y, weights)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics - keep on GPU
        loss_accumulator.append(loss.detach())
        pred = logits.argmax(dim=1)  # argmax works on logits too

        # Track decomposed losses for DisCo
        if track_decomposed:
            decomposed = loss_fn.get_decomposed_losses()
            ce_loss_accumulator.append(decomposed['ce_loss'])
            disco_term_accumulator.append(decomposed['disco_term'])

        # Collect predictions and labels for group-balanced accuracy
        # Use .detach() to prevent gradient graph buildup, keep on GPU
        all_predictions.append(pred.detach())
        all_labels.append(batch.y)
        all_weights.append(weights)

    # Calculate total loss (single GPU-CPU sync at end)
    total_loss = torch.stack(loss_accumulator).sum().item()

    # Update scheduler
    if hasattr(scheduler, 'step'):
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(total_loss)
        else:
            scheduler.step()

    # Concatenate on GPU, then move to CPU for accuracy calculation
    all_predictions = torch.cat(all_predictions).cpu()
    all_labels = torch.cat(all_labels).cpu()
    all_weights = torch.cat(all_weights).cpu()

    accuracy = calculate_group_balanced_accuracy(
        all_predictions, all_labels, all_weights, use_groups, num_classes
    )

    avg_loss = total_loss / len(train_loader)

    # Return decomposed losses if tracking DisCo
    if track_decomposed and ce_loss_accumulator:
        avg_ce_loss = sum(ce_loss_accumulator) / len(ce_loss_accumulator)
        avg_disco_term = sum(disco_term_accumulator) / len(disco_term_accumulator)
        return avg_loss, accuracy, {'ce_loss': avg_ce_loss, 'disco_term': avg_disco_term}

    return avg_loss, accuracy, None


def evaluate_model(model, data_loader, loss_fn, device, use_groups=False, num_classes=4,
                   loss_requires_mass=False):
    """
    Evaluate model on validation or test set.

    Args:
        model: PyTorch model to evaluate
        data_loader: Data loader for evaluation
        loss_fn: Loss function
        device: Device for evaluation
        use_groups: Whether using grouped backgrounds
        num_classes: Number of classes
        loss_requires_mass: Whether loss function needs mass tensors (DisCo)

    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()

    # Accumulate losses on GPU to avoid per-batch synchronization
    loss_accumulator = []

    # Accumulate decomposed losses for DisCo tracking
    ce_loss_accumulator = []
    disco_term_accumulator = []
    track_decomposed = loss_requires_mass and hasattr(loss_fn, 'get_decomposed_losses')

    # Collect all predictions, labels, and weights for group-balanced accuracy
    # Keep on GPU during evaluation to avoid unnecessary CPU transfers
    all_predictions = []
    all_labels = []
    all_weights = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)

            # Extract weights for loss function
            weights = extract_weights_from_batch(batch)

            # Compute loss (with or without mass for DisCo)
            if loss_requires_mass:
                mass1 = batch.mass1.squeeze()
                mass2 = batch.mass2.squeeze()
                loss = loss_fn(logits, batch.y, weights, mass1, mass2)
            else:
                loss = loss_fn(logits, batch.y, weights)

            # Statistics - keep on GPU
            loss_accumulator.append(loss)
            pred = logits.argmax(dim=1)  # argmax works on logits too

            # Track decomposed losses for DisCo
            if track_decomposed:
                decomposed = loss_fn.get_decomposed_losses()
                ce_loss_accumulator.append(decomposed['ce_loss'])
                disco_term_accumulator.append(decomposed['disco_term'])

            # Collect predictions and labels for group-balanced accuracy
            # Keep on GPU to avoid per-batch CPU transfers
            all_predictions.append(pred)
            all_labels.append(batch.y)
            all_weights.append(weights)

    # Calculate total loss (single GPU-CPU sync at end)
    total_loss = torch.stack(loss_accumulator).sum().item()

    # Concatenate on GPU, then move to CPU for accuracy calculation
    all_predictions = torch.cat(all_predictions).cpu()
    all_labels = torch.cat(all_labels).cpu()
    all_weights = torch.cat(all_weights).cpu()

    accuracy = calculate_group_balanced_accuracy(
        all_predictions, all_labels, all_weights, use_groups, num_classes
    )

    avg_loss = total_loss / len(data_loader)

    # Return decomposed losses if tracking DisCo
    if track_decomposed and ce_loss_accumulator:
        avg_ce_loss = sum(ce_loss_accumulator) / len(ce_loss_accumulator)
        avg_disco_term = sum(disco_term_accumulator) / len(disco_term_accumulator)
        return avg_loss, accuracy, {'ce_loss': avg_ce_loss, 'disco_term': avg_disco_term}

    return avg_loss, accuracy, None


def get_detailed_predictions(model, data_loader, device, use_groups=False, num_classes=4):
    """
    Get detailed predictions for logging and analysis.

    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device for inference
        use_groups: Whether using grouped backgrounds
        num_classes: Number of classes

    Returns:
        Tuple of (predictions, labels, weights) tensors
    """
    model.eval()
    all_predictions = []
    all_labels = []
    all_weights = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)
            pred = out.argmax(dim=1)
            weights = extract_weights_from_batch(batch)

            # Keep on GPU to avoid per-batch CPU transfers
            all_predictions.append(pred)
            all_labels.append(batch.y)
            all_weights.append(weights)

    # Concatenate on GPU, then move to CPU once at the end
    return (torch.cat(all_predictions).cpu(),
            torch.cat(all_labels).cpu(),
            torch.cat(all_weights).cpu())


class PerformanceMonitor:
    """Monitor system performance during training."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.process = psutil.Process()

    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        memory_info = self.process.memory_info()
        return {
            'memory_mb': memory_info.rss / (1024 * 1024),  # Convert to MB
            'cpu_percent': self.process.cpu_percent(),
            'model_type': self.model_type
        }


def create_optimizer(optimizer_type: str, model_parameters, learning_rate: float,
                    weight_decay: float):
    """
    Create optimizer based on configuration.

    Args:
        optimizer_type: Type of optimizer (Adam, RMSprop, Adadelta)
        model_parameters: Model parameters to optimize
        learning_rate: Initial learning rate
        weight_decay: Weight decay factor

    Returns:
        Configured optimizer instance
    """
    if optimizer_type == "RMSprop":
        return torch.optim.RMSprop(model_parameters, lr=learning_rate,
                                  momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == "Adam":
        return torch.optim.Adam(model_parameters, lr=learning_rate,
                               weight_decay=weight_decay)
    elif optimizer_type == "Adadelta":
        return torch.optim.Adadelta(model_parameters, lr=learning_rate,
                                   weight_decay=weight_decay)
    else:
        raise NotImplementedError(f"Unsupported optimizer {optimizer_type}")


def create_scheduler(scheduler_type: str, optimizer, learning_rate: float):
    """
    Create learning rate scheduler based on configuration.

    Args:
        scheduler_type: Type of scheduler
        optimizer: Optimizer instance
        learning_rate: Base learning rate

    Returns:
        Configured scheduler instance
    """
    if scheduler_type == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    elif scheduler_type == "ExponentialLR":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    elif scheduler_type == "CyclicLR":
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=learning_rate/5., max_lr=learning_rate*2,
            step_size_up=3, step_size_down=5, cycle_momentum=False
        )
    elif scheduler_type == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    else:
        raise NotImplementedError(f"Unsupported scheduler {scheduler_type}")


def setup_device(device_str: str) -> torch.device:
    """
    Setup and configure training device.

    Args:
        device_str: Device string ('cpu' or 'cuda')

    Returns:
        Configured torch device
    """
    device = torch.device(device_str)
    if device.type == "cuda":
        logging.info("Using CUDA")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        logging.info("Using CPU")

    return device