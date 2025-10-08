#!/usr/bin/env python
"""
Export trained ParticleNet models to TorchScript for C++ inference.

This script exports multiclass ParticleNet models to TorchScript format using
torch.jit.trace() for compatibility with SKNanoAnalyzer C++ framework.

KEY FIX: Uses CPPCompatibleParticleNet wrapper to avoid torch_cluster dependency
by passing pre-computed edge_index to all DynamicEdgeConv layers.

Features:
- Loads trained model checkpoints and exports to TorchScript
- Uses tracing (not scripting) for PyTorch Geometric compatibility
- Validates model architecture matches training configuration
- Supports batch export of multiple models
- Creates TorchScript models that accept pre-computed edge indices
- NO torch_cluster dependency in exported models

Usage:
    # Export single model
    python export_to_torchscript.py --channel Run1E2Mu --signal MHc100_MA95 --fold 3

    # Export all signals for a channel
    python export_to_torchscript.py --channel Run1E2Mu --all-signals --fold 3

    # Export everything
    python export_to_torchscript.py --all --fold 3

Author: Claude Code + Human
Date: 2025-10-08
Last Updated: 2025-10-08 (fixed torch_cluster dependency issue)
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph, global_mean_pool
from MultiClassModels import MultiClassParticleNet
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CPPCompatibleParticleNet(nn.Module):
    """
    Wrapper around MultiClassParticleNet that uses pre-computed edge_index for all layers.

    This avoids torch_cluster dependency by reusing the input edge_index instead of
    recomputing k-NN graphs dynamically in each DynamicEdgeConv layer.

    The original MultiClassParticleNet calls:
    - conv1(x, edge_index, batch)  ✓ uses provided edge_index
    - conv2(conv1, batch)           ✗ recomputes edge_index with torch_cluster
    - conv3(conv2, batch)           ✗ recomputes edge_index with torch_cluster

    This wrapper fixes conv2 and conv3 to use the provided edge_index.
    """

    def __init__(self, original_model):
        super().__init__()

        # Copy all layers from original model
        self.gn0 = original_model.gn0
        self.conv1 = original_model.conv1
        self.conv2 = original_model.conv2
        self.conv3 = original_model.conv3
        self.bn0 = original_model.bn0
        self.dense1 = original_model.dense1
        self.bn1 = original_model.bn1
        self.dense2 = original_model.dense2
        self.bn2 = original_model.bn2
        self.output = original_model.output
        self.dropout_p = original_model.dropout_p
        self.num_classes = original_model.num_classes

    def forward(self, x, edge_index, graph_input, batch=None):
        """
        Forward pass using pre-computed edge_index for all conv layers.

        Args:
            x: Node features (N_nodes, 9)
            edge_index: Pre-computed k-NN graph (2, N_edges)
            graph_input: Era encoding (1, 4)
            batch: Batch assignment (N_nodes,)

        Returns:
            Logits (1, 4) for 4-class classification
        """
        # Input normalization
        x = self.gn0(x, batch=batch)

        # Graph convolution layers - ALL use the SAME edge_index
        conv1 = self.conv1(x, edge_index, batch=batch)
        conv2 = self.conv2(conv1, edge_index, batch=batch)  # FIX: pass edge_index
        conv3 = self.conv3(conv2, edge_index, batch=batch)  # FIX: pass edge_index
        x = torch.cat([conv1, conv2, conv3], dim=1)

        # Global pooling
        x = global_mean_pool(x, batch=batch)
        x = torch.cat([x, graph_input], dim=1)
        x = self.bn0(x)

        # Dense classification layers
        x = F.leaky_relu(self.dense1(x))
        x = self.bn1(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = F.leaky_relu(self.dense2(x))
        x = self.bn2(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.output(x)

        return x


# Model configuration from training
MODEL_CONFIG = {
    'num_node_features': 9,    # E, Px, Py, Pz, Charge, IsMuon, IsElectron, IsJet, IsBjet
    'num_graph_features': 4,   # Era encoding (one-hot 4D)
    'num_classes': 4,          # Signal + 3 backgrounds (nonprompt, diboson, ttZ)
    'num_hidden': 128,         # Hidden layer size
    'dropout_p': 0.25          # Dropout probability
}

# Signal mass points for each channel
SIGNALS = {
    'Run1E2Mu': [
        'TTToHcToWAToMuMu-MHc160_MA85',
        'TTToHcToWAToMuMu-MHc130_MA90',
        'TTToHcToWAToMuMu-MHc100_MA95'
    ],
    'Run3Mu': [
        'TTToHcToWAToMuMu-MHc160_MA85',
        'TTToHcToWAToMuMu-MHc130_MA90',
        'TTToHcToWAToMuMu-MHc100_MA95'
    ]
}


def find_model_checkpoint(channel: str, signal: str, fold: int, base_dir: str = 'results_bjets') -> Path:
    """
    Find the model checkpoint file for given channel/signal/fold.

    Args:
        channel: Channel name (Run1E2Mu or Run3Mu)
        signal: Signal name (e.g., MHc100_MA95 or full name)
        fold: Fold number
        base_dir: Base results directory

    Returns:
        Path to checkpoint file

    Raises:
        FileNotFoundError: If checkpoint not found
    """
    # Handle both short and full signal names
    if not signal.startswith('TTToHcToWAToMuMu'):
        signal_full = f'TTToHcToWAToMuMu-{signal}'
    else:
        signal_full = signal

    model_dir = Path(base_dir) / channel / 'multiclass' / signal_full / f'fold-{fold}' / 'models'

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Find .pt file (should be exactly one)
    pt_files = list(model_dir.glob('*.pt'))
    pt_files = [f for f in pt_files if not f.name.endswith('_scripted.pt')]

    if not pt_files:
        raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    if len(pt_files) > 1:
        logger.warning(f"Multiple checkpoints found in {model_dir}, using {pt_files[0]}")

    return pt_files[0]


def load_model_checkpoint(checkpoint_path: Path) -> dict:
    """
    Load PyTorch checkpoint containing model state dict.

    Args:
        checkpoint_path: Path to .pt checkpoint file

    Returns:
        Checkpoint dictionary
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        raise

    # Validate checkpoint structure
    required_keys = ['model_state_dict']
    for key in required_keys:
        if key not in checkpoint:
            raise ValueError(f"Checkpoint missing required key: {key}")

    logger.info(f"Checkpoint loaded successfully")
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"  Val loss: {checkpoint.get('val_loss', 'unknown')}")
    logger.info(f"  Best loss: {checkpoint.get('best_loss', 'unknown')}")

    return checkpoint


def create_model(config: dict) -> MultiClassParticleNet:
    """
    Create MultiClassParticleNet model with specified configuration.

    Args:
        config: Model configuration dictionary

    Returns:
        Initialized MultiClassParticleNet model
    """
    model = MultiClassParticleNet(
        num_node_features=config['num_node_features'],
        num_graph_features=config['num_graph_features'],
        num_classes=config['num_classes'],
        num_hidden=config['num_hidden'],
        dropout_p=config['dropout_p']
    )

    logger.info(f"Created MultiClassParticleNet:")
    logger.info(f"  Node features: {config['num_node_features']}")
    logger.info(f"  Graph features: {config['num_graph_features']}")
    logger.info(f"  Classes: {config['num_classes']}")
    logger.info(f"  Hidden units: {config['num_hidden']}")
    logger.info(f"  Dropout: {config['dropout_p']}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    return model


def create_example_input(config: dict, num_nodes: int = 20, seed: int = 42):
    """
    Create example input tensors for tracing.

    Args:
        config: Model configuration
        num_nodes: Number of nodes in example graph
        seed: Random seed for reproducibility

    Returns:
        Tuple of (x, edge_index, graph_input, batch)
    """
    torch.manual_seed(seed)

    # Node features: (N_nodes, num_node_features)
    x = torch.randn(num_nodes, config['num_node_features'])

    # Batch assignment (single graph)
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Edge index from k-NN graph (k=4)
    edge_index = knn_graph(x, k=4, batch=batch, loop=False)

    # Graph-level features: (1, num_graph_features)
    graph_input = torch.randn(1, config['num_graph_features'])

    logger.info(f"Created example input:")
    logger.info(f"  x: {x.shape}")
    logger.info(f"  edge_index: {edge_index.shape}")
    logger.info(f"  graph_input: {graph_input.shape}")
    logger.info(f"  batch: {batch.shape}")

    return x, edge_index, graph_input, batch


def export_to_torchscript(model: nn.Module, example_inputs: tuple, output_path: Path):
    """
    Export model to TorchScript using tracing.

    Args:
        model: PyTorch model to export
        example_inputs: Tuple of example inputs for tracing
        output_path: Path to save TorchScript model
    """
    logger.info("Exporting model to TorchScript...")

    # Set model to eval mode
    model.eval()

    # Verify model works with example inputs
    try:
        with torch.no_grad():
            test_output = model(*example_inputs)
        logger.info(f"  Test forward pass output shape: {test_output.shape}")
    except Exception as e:
        logger.error(f"Model forward pass failed: {e}")
        raise

    # Trace the model
    try:
        traced_model = torch.jit.trace(model, example_inputs)
        logger.info("  Model traced successfully")
    except Exception as e:
        logger.error(f"Tracing failed: {e}")
        raise

    # Verify traced model produces same output
    try:
        with torch.no_grad():
            traced_output = traced_model(*example_inputs)

        max_diff = torch.abs(test_output - traced_output).max().item()
        logger.info(f"  Verification max diff: {max_diff:.2e}")

        if max_diff > 1e-5:
            logger.warning(f"Traced model output differs by {max_diff:.2e}")
    except Exception as e:
        logger.error(f"Traced model verification failed: {e}")
        raise

    # Save TorchScript model
    try:
        traced_model.save(str(output_path))
        logger.info(f"  Saved TorchScript model to {output_path}")

        # Verify saved model can be loaded
        loaded_model = torch.jit.load(str(output_path))
        with torch.no_grad():
            loaded_output = loaded_model(*example_inputs)

        reload_diff = torch.abs(test_output - loaded_output).max().item()
        logger.info(f"  Reload verification max diff: {reload_diff:.2e}")

    except Exception as e:
        logger.error(f"Saving TorchScript model failed: {e}")
        raise

    # Report file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  TorchScript model size: {file_size_mb:.2f} MB")


def export_model(channel: str, signal: str, fold: int, base_dir: str = 'results_bjets'):
    """
    Export a single model to TorchScript.

    Args:
        channel: Channel name
        signal: Signal name
        fold: Fold number
        base_dir: Base results directory
    """
    logger.info("=" * 60)
    logger.info(f"Exporting model: {channel} / {signal} / fold-{fold}")
    logger.info("=" * 60)

    # Find checkpoint
    checkpoint_path = find_model_checkpoint(channel, signal, fold, base_dir)

    # Load checkpoint
    checkpoint = load_model_checkpoint(checkpoint_path)

    # Create model
    model = create_model(MODEL_CONFIG)

    # Load state dict
    logger.info("Loading model state dict...")
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("  State dict loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load state dict: {e}")
        raise

    # Wrap with C++-compatible version that uses pre-computed edge_index
    logger.info("Creating C++-compatible wrapper...")
    cpp_model = CPPCompatibleParticleNet(model)
    logger.info("  Wrapper created - all conv layers will use pre-computed edge_index")

    # Create example input
    example_inputs = create_example_input(MODEL_CONFIG)

    # Generate output path
    output_path = checkpoint_path.parent / f"{checkpoint_path.stem}_scripted.pt"

    # Export to TorchScript
    export_to_torchscript(cpp_model, example_inputs, output_path)

    logger.info("=" * 60)
    logger.info("Export completed successfully!")
    logger.info("=" * 60)

    return output_path


def main():
    """Main entry point for TorchScript export."""
    parser = argparse.ArgumentParser(
        description='Export ParticleNet models to TorchScript format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export single model
  python export_to_torchscript.py --channel Run1E2Mu --signal MHc100_MA95 --fold 3

  # Export all signals for a channel
  python export_to_torchscript.py --channel Run1E2Mu --all-signals --fold 3

  # Export everything
  python export_to_torchscript.py --all --fold 3
        """
    )

    parser.add_argument('--channel', type=str, choices=['Run1E2Mu', 'Run3Mu'],
                       help='Channel name (required unless --all)')
    parser.add_argument('--signal', type=str,
                       help='Signal name (e.g., MHc100_MA95 or full name)')
    parser.add_argument('--fold', type=int, default=3,
                       help='Fold number (default: 3)')
    parser.add_argument('--all-signals', action='store_true',
                       help='Export all signals for specified channel')
    parser.add_argument('--all', action='store_true',
                       help='Export all channels and signals')
    parser.add_argument('--base-dir', type=str, default='results_bjets',
                       help='Base results directory (default: results_bjets)')

    args = parser.parse_args()

    # Validate arguments
    if not args.all:
        if not args.channel:
            parser.error("--channel required unless --all is specified")
        if not args.all_signals and not args.signal:
            parser.error("--signal required unless --all-signals or --all is specified")

    # Determine what to export
    export_tasks = []

    if args.all:
        # Export everything
        for channel in SIGNALS.keys():
            for signal in SIGNALS[channel]:
                export_tasks.append((channel, signal, args.fold))
    elif args.all_signals:
        # Export all signals for specified channel
        for signal in SIGNALS[args.channel]:
            export_tasks.append((args.channel, signal, args.fold))
    else:
        # Export single model
        export_tasks.append((args.channel, args.signal, args.fold))

    # Execute exports
    logger.info(f"Starting export of {len(export_tasks)} model(s)...")

    success_count = 0
    failed_models = []

    for channel, signal, fold in export_tasks:
        try:
            output_path = export_model(channel, signal, fold, args.base_dir)
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to export {channel}/{signal}/fold-{fold}: {e}")
            failed_models.append((channel, signal, fold))

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXPORT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total models: {len(export_tasks)}")
    logger.info(f"Successful: {success_count}")
    logger.info(f"Failed: {len(failed_models)}")

    if failed_models:
        logger.info("\nFailed models:")
        for channel, signal, fold in failed_models:
            logger.info(f"  - {channel}/{signal}/fold-{fold}")
        sys.exit(1)
    else:
        logger.info("\nAll models exported successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
