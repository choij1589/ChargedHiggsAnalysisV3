#!/usr/bin/env python
"""
Machine Learning Tools for ParticleNet Training.

This module provides utility classes for training management including:
- EarlyStopper: Implements early stopping to prevent overfitting
- SummaryWriter: Handles training metrics logging and visualization
"""

import os
import logging
import json
import time
from typing import Dict, Any, Optional, List
import torch
import numpy as np


class EarlyStopper:
    """
    Early stopping utility to prevent overfitting during training.
    
    Monitors validation loss and stops training when no improvement is seen
    for a specified number of epochs (patience).
    """
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, 
                 restore_best_weights: bool = True, path: Optional[str] = None):
        """
        Initialize early stopper.
        
        Args:
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best model weights
            path: Path to save the best model checkpoint
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.path = path
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        logging.info(f"EarlyStopper initialized: patience={patience}, min_delta={min_delta}")
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should be stopped.
        
        Args:
            val_loss: Current validation loss
            model: The model being trained
            
        Returns:
            True if training should be stopped, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
                
            # Save best model if path is provided
            if self.path:
                self._save_checkpoint(model, val_loss)
                
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            logging.info(f"Early stopping triggered after {self.counter} epochs without improvement")
            
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logging.info("Restored best model weights")
                
        return self.early_stop
    
    def _save_checkpoint(self, model: torch.nn.Module, val_loss: float) -> None:
        """Save model checkpoint."""
        if self.path:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'best_loss': self.best_loss,
                'epoch': self.counter
            }
            torch.save(checkpoint, self.path)
            logging.debug(f"Checkpoint saved to {self.path}")
    
    def reset(self) -> None:
        """Reset early stopper state."""
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        logging.info("EarlyStopper reset")


class SummaryWriter:
    """
    Training metrics summary writer for logging and visualization.
    
    Handles logging of training metrics, loss curves, and performance statistics.
    """
    
    def __init__(self, name: str, log_dir: str = "logs", save_csv: bool = True):
        """
        Initialize summary writer.
        
        Args:
            name: Name identifier for this training run
            log_dir: Directory to save logs
            save_csv: Whether to save metrics to CSV files
        """
        self.name = name
        self.log_dir = log_dir
        self.save_csv = save_csv
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics = {
            'train_loss': [],
            'valid_loss': [],
            'train_acc': [],
            'valid_acc': [],
            'epoch': [],
            'timestamp': []
        }
        
        # Training start time
        self.start_time = time.time()
        
        logging.info(f"SummaryWriter initialized: {name}, log_dir={log_dir}")
    
    def log_metrics(self, epoch: int, train_loss: float, valid_loss: float,
                   train_acc: float, valid_acc: float) -> None:
        """
        Log training metrics for an epoch.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            valid_loss: Validation loss
            train_acc: Training accuracy
            valid_acc: Validation accuracy
        """
        current_time = time.time()
        
        # Store metrics
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['valid_loss'].append(valid_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['valid_acc'].append(valid_acc)
        self.metrics['timestamp'].append(current_time)
        
        # Log to console
        elapsed_time = current_time - self.start_time
        logging.info(f"Epoch {epoch:3d} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Valid Loss: {valid_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Valid Acc: {valid_acc:.4f} | "
                    f"Time: {elapsed_time:.1f}s")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters used for training.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        hyperparams['training_start_time'] = time.strftime('%Y-%m-%d %H:%M:%S', 
                                                          time.localtime(self.start_time))
        
        hyperparams_path = os.path.join(self.log_dir, f"{self.name}_hyperparams.json")
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=2)
        
        logging.info(f"Hyperparameters saved to {hyperparams_path}")
    
    def save_metrics_csv(self) -> str:
        """
        Save metrics to CSV file.
        
        Returns:
            Path to the saved CSV file
        """
        if not self.save_csv:
            return None
            
        import pandas as pd
        
        # Create DataFrame from metrics
        df = pd.DataFrame(self.metrics)
        
        # Add derived metrics
        df['train_loss_smooth'] = df['train_loss'].rolling(window=5, center=True).mean()
        df['valid_loss_smooth'] = df['valid_loss'].rolling(window=5, center=True).mean()
        
        # Save to CSV
        csv_path = os.path.join(self.log_dir, f"{self.name}_metrics.csv")
        df.to_csv(csv_path, index=False)
        
        logging.info(f"Metrics saved to {csv_path}")
        return csv_path
    
    def get_best_epoch(self) -> int:
        """
        Get the epoch with the best validation loss.
        
        Returns:
            Epoch number with best validation loss
        """
        if not self.metrics['valid_loss']:
            return 0
            
        best_idx = np.argmin(self.metrics['valid_loss'])
        return self.metrics['epoch'][best_idx]
    
    def get_best_metrics(self) -> Dict[str, float]:
        """
        Get the best metrics achieved during training.
        
        Returns:
            Dictionary with best metrics
        """
        if not self.metrics['valid_loss']:
            return {}
            
        best_idx = np.argmin(self.metrics['valid_loss'])
        
        return {
            'best_epoch': self.metrics['epoch'][best_idx],
            'best_train_loss': self.metrics['train_loss'][best_idx],
            'best_valid_loss': self.metrics['valid_loss'][best_idx],
            'best_train_acc': self.metrics['train_acc'][best_idx],
            'best_valid_acc': self.metrics['valid_acc'][best_idx]
        }
    
    def plot_metrics(self, save_path: Optional[str] = None) -> None:
        """
        Plot training metrics.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            epochs = self.metrics['epoch']
            
            # Plot losses
            ax1.plot(epochs, self.metrics['train_loss'], 'b-', label='Training Loss', alpha=0.7)
            ax1.plot(epochs, self.metrics['valid_loss'], 'r-', label='Validation Loss', alpha=0.7)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title(f'Training Progress - {self.name}')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot accuracies
            ax2.plot(epochs, self.metrics['train_acc'], 'b-', label='Training Accuracy', alpha=0.7)
            ax2.plot(epochs, self.metrics['valid_acc'], 'r-', label='Validation Accuracy', alpha=0.7)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Training plot saved to {save_path}")
            else:
                plt.show()
                
            plt.close()
            
        except ImportError:
            logging.warning("Matplotlib not available for plotting")
    
    def close(self) -> None:
        """Close the summary writer and save final metrics."""
        if self.save_csv:
            self.save_metrics_csv()
        
        # Log final summary
        best_metrics = self.get_best_metrics()
        if best_metrics:
            logging.info("=" * 60)
            logging.info("TRAINING SUMMARY")
            logging.info("=" * 60)
            logging.info(f"Best Epoch: {best_metrics['best_epoch']}")
            logging.info(f"Best Validation Loss: {best_metrics['best_valid_loss']:.4f}")
            logging.info(f"Best Validation Accuracy: {best_metrics['best_valid_acc']:.4f}")
            logging.info(f"Total Training Time: {time.time() - self.start_time:.1f} seconds")
            logging.info("=" * 60)


def create_early_stopper(patience: int = 7, path: Optional[str] = None) -> EarlyStopper:
    """
    Factory function to create EarlyStopper instance.
    
    Args:
        patience: Number of epochs to wait for improvement
        path: Path to save best model checkpoint
        
    Returns:
        EarlyStopper instance
    """
    return EarlyStopper(patience=patience, path=path)


def create_summary_writer(name: str, log_dir: str = "logs") -> SummaryWriter:
    """
    Factory function to create SummaryWriter instance.
    
    Args:
        name: Name identifier for training run
        log_dir: Directory to save logs
        
    Returns:
        SummaryWriter instance
    """
    return SummaryWriter(name=name, log_dir=log_dir)
