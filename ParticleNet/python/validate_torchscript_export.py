#!/usr/bin/env python
"""
Validate TorchScript exported ParticleNet models.

This script compares outputs between original PyTorch models and their
TorchScript-exported counterparts to ensure numerical equivalence.

Features:
- Loads both original checkpoint and TorchScript model
- Creates identical test inputs with fixed random seed
- Compares outputs with configurable tolerances
- Reports detailed numerical differences
- Supports batch validation of multiple models

Usage:
    # Validate single model
    python validate_torchscript_export.py --model results_bjets/Run1E2Mu/.../ParticleNet-*_scripted.pt

    # Validate all models in directory
    python validate_torchscript_export.py --all --fold 3

Author: Claude Code + Human
Date: 2025-10-08
"""

import argparse
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.nn import knn_graph
from MultiClassModels import MultiClassParticleNet
import numpy as np
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configuration (must match training)
MODEL_CONFIG = {
    'num_node_features': 9,    # E, Px, Py, Pz, Charge, IsMuon, IsElectron, IsJet, IsBjet
    'num_graph_features': 4,   # Era encoding (one-hot 4D)
    'num_classes': 4,          # Signal + 3 backgrounds
    'num_hidden': 128,
    'dropout_p': 0.25
}

# Validation tolerances
RTOL = 1e-5  # Relative tolerance
ATOL = 1e-6  # Absolute tolerance


def find_original_checkpoint(scripted_model_path: Path) -> Path:
    """
    Find the original checkpoint corresponding to a TorchScript model.

    Args:
        scripted_model_path: Path to TorchScript model (*_scripted.pt)

    Returns:
        Path to original checkpoint

    Raises:
        FileNotFoundError: If original checkpoint not found
    """
    # Remove _scripted suffix to get original name
    if not scripted_model_path.name.endswith('_scripted.pt'):
        raise ValueError(f"Expected *_scripted.pt file, got {scripted_model_path.name}")

    original_name = scripted_model_path.name.replace('_scripted.pt', '.pt')
    original_path = scripted_model_path.parent / original_name

    if not original_path.exists():
        raise FileNotFoundError(f"Original checkpoint not found: {original_path}")

    return original_path


def load_original_model(checkpoint_path: Path) -> nn.Module:
    """
    Load original PyTorch model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded PyTorch model in eval mode
    """
    logger.info(f"Loading original checkpoint: {checkpoint_path.name}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    model = MultiClassParticleNet(
        num_node_features=MODEL_CONFIG['num_node_features'],
        num_graph_features=MODEL_CONFIG['num_graph_features'],
        num_classes=MODEL_CONFIG['num_classes'],
        num_hidden=MODEL_CONFIG['num_hidden'],
        dropout_p=MODEL_CONFIG['dropout_p']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info("  Original model loaded successfully")
    return model


def load_torchscript_model(scripted_path: Path) -> torch.jit.ScriptModule:
    """
    Load TorchScript model.

    Args:
        scripted_path: Path to TorchScript model

    Returns:
        Loaded TorchScript model
    """
    logger.info(f"Loading TorchScript model: {scripted_path.name}")

    try:
        model = torch.jit.load(str(scripted_path), map_location='cpu')
        model.eval()
        logger.info("  TorchScript model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load TorchScript model: {e}")
        raise


def create_test_inputs(config: dict, num_nodes: int = 20, seed: int = 42):
    """
    Create deterministic test inputs.

    Args:
        config: Model configuration
        num_nodes: Number of nodes
        seed: Random seed for reproducibility

    Returns:
        Tuple of (x, edge_index, graph_input, batch)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Node features
    x = torch.randn(num_nodes, config['num_node_features'])

    # Batch assignment
    batch = torch.zeros(num_nodes, dtype=torch.long)

    # Edge index from k-NN
    edge_index = knn_graph(x, k=4, batch=batch, loop=False)

    # Graph features
    graph_input = torch.randn(1, config['num_graph_features'])

    return x, edge_index, graph_input, batch


def compare_outputs(output_original: torch.Tensor, output_scripted: torch.Tensor,
                   rtol: float = RTOL, atol: float = ATOL) -> dict:
    """
    Compare outputs from original and TorchScript models.

    Args:
        output_original: Output from original PyTorch model
        output_scripted: Output from TorchScript model
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        Dictionary with comparison results
    """
    # Convert to numpy for detailed analysis
    out_orig_np = output_original.detach().cpu().numpy()
    out_script_np = output_scripted.detach().cpu().numpy()

    # Compute differences
    abs_diff = np.abs(out_orig_np - out_script_np)
    rel_diff = abs_diff / (np.abs(out_orig_np) + 1e-10)  # Avoid division by zero

    # Check if outputs match within tolerances
    matches = np.allclose(out_orig_np, out_script_np, rtol=rtol, atol=atol)

    results = {
        'matches': matches,
        'max_abs_diff': np.max(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'max_rel_diff': np.max(rel_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'shape_match': output_original.shape == output_scripted.shape,
        'original_shape': output_original.shape,
        'scripted_shape': output_scripted.shape
    }

    return results


def validate_model(scripted_model_path: Path, num_tests: int = 5) -> bool:
    """
    Validate a single TorchScript model against its original.

    Args:
        scripted_model_path: Path to TorchScript model
        num_tests: Number of random test inputs to validate

    Returns:
        True if validation passed, False otherwise
    """
    logger.info("=" * 60)
    logger.info(f"Validating: {scripted_model_path.name}")
    logger.info("=" * 60)

    try:
        # Find and load original model
        original_checkpoint_path = find_original_checkpoint(scripted_model_path)
        original_model = load_original_model(original_checkpoint_path)

        # Load TorchScript model
        scripted_model = load_torchscript_model(scripted_model_path)

        # Run validation with multiple test inputs
        all_passed = True
        results_summary = []

        for test_idx in range(num_tests):
            logger.info(f"\nTest {test_idx + 1}/{num_tests}:")

            # Create test inputs with different seeds
            test_inputs = create_test_inputs(MODEL_CONFIG, num_nodes=20, seed=42 + test_idx)
            x, edge_index, graph_input, batch = test_inputs

            logger.info(f"  Input shapes: x={x.shape}, edge_index={edge_index.shape}, "
                       f"graph_input={graph_input.shape}, batch={batch.shape}")

            # Run inference on both models
            with torch.no_grad():
                output_original = original_model(x, edge_index, graph_input, batch)
                output_scripted = scripted_model(x, edge_index, graph_input, batch)

            # Compare outputs
            results = compare_outputs(output_original, output_scripted)
            results_summary.append(results)

            # Log results
            logger.info(f"  Shape match: {results['shape_match']}")
            logger.info(f"  Max abs diff: {results['max_abs_diff']:.2e}")
            logger.info(f"  Mean abs diff: {results['mean_abs_diff']:.2e}")
            logger.info(f"  Max rel diff: {results['max_rel_diff']:.2e}")
            logger.info(f"  Mean rel diff: {results['mean_rel_diff']:.2e}")

            if results['matches']:
                logger.info(f"  ✅ PASSED (within rtol={RTOL:.0e}, atol={ATOL:.0e})")
            else:
                logger.warning(f"  ❌ FAILED (exceeds rtol={RTOL:.0e}, atol={ATOL:.0e})")
                all_passed = False

        # Overall summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 60)

        max_abs_diff_overall = max(r['max_abs_diff'] for r in results_summary)
        mean_abs_diff_overall = np.mean([r['mean_abs_diff'] for r in results_summary])
        max_rel_diff_overall = max(r['max_rel_diff'] for r in results_summary)

        logger.info(f"Tests run: {num_tests}")
        logger.info(f"Tests passed: {sum(r['matches'] for r in results_summary)}")
        logger.info(f"Tests failed: {sum(not r['matches'] for r in results_summary)}")
        logger.info(f"\nOverall statistics:")
        logger.info(f"  Max absolute difference: {max_abs_diff_overall:.2e}")
        logger.info(f"  Mean absolute difference: {mean_abs_diff_overall:.2e}")
        logger.info(f"  Max relative difference: {max_rel_diff_overall:.2e}")

        if all_passed:
            logger.info(f"\n✅ VALIDATION PASSED")
            logger.info(f"   TorchScript model produces numerically equivalent outputs")
        else:
            logger.warning(f"\n❌ VALIDATION FAILED")
            logger.warning(f"   TorchScript model outputs differ from original")

        logger.info("=" * 60)

        return all_passed

    except Exception as e:
        logger.error(f"Validation failed with error: {e}")
        logger.error("=" * 60)
        return False


def find_all_scripted_models(base_dir: str = 'results_bjets', fold: int = 3) -> list:
    """
    Find all TorchScript models in the results directory.

    Args:
        base_dir: Base results directory
        fold: Fold number

    Returns:
        List of paths to TorchScript models
    """
    base_path = Path(base_dir)
    pattern = f"*/multiclass/*/fold-{fold}/models/*_scripted.pt"
    scripted_models = list(base_path.glob(pattern))

    logger.info(f"Found {len(scripted_models)} TorchScript models for fold-{fold}")
    return sorted(scripted_models)


def main():
    """Main entry point for validation."""
    parser = argparse.ArgumentParser(
        description='Validate TorchScript ParticleNet models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single model
  python validate_torchscript_export.py --model results_bjets/Run1E2Mu/.../ParticleNet-*_scripted.pt

  # Validate all models
  python validate_torchscript_export.py --all --fold 3

  # Validate with more test cases
  python validate_torchscript_export.py --all --num-tests 10
        """
    )

    parser.add_argument('--model', type=str,
                       help='Path to TorchScript model to validate')
    parser.add_argument('--all', action='store_true',
                       help='Validate all TorchScript models')
    parser.add_argument('--fold', type=int, default=3,
                       help='Fold number for --all mode (default: 3)')
    parser.add_argument('--base-dir', type=str, default='results_bjets',
                       help='Base results directory (default: results_bjets)')
    parser.add_argument('--num-tests', type=int, default=5,
                       help='Number of test inputs per model (default: 5)')
    parser.add_argument('--rtol', type=float, default=RTOL,
                       help=f'Relative tolerance (default: {RTOL:.0e})')
    parser.add_argument('--atol', type=float, default=ATOL,
                       help=f'Absolute tolerance (default: {ATOL:.0e})')

    args = parser.parse_args()

    # Use tolerances from args (update module-level constants locally)
    rtol = args.rtol
    atol = args.atol

    # Determine models to validate
    if args.all:
        model_paths = find_all_scripted_models(args.base_dir, args.fold)
        if not model_paths:
            logger.error(f"No TorchScript models found in {args.base_dir} for fold-{args.fold}")
            sys.exit(1)
    elif args.model:
        model_paths = [Path(args.model)]
    else:
        parser.error("Either --model or --all must be specified")

    # Validate models
    logger.info(f"Validating {len(model_paths)} model(s)...")
    logger.info(f"Tolerance: rtol={rtol:.0e}, atol={atol:.0e}")
    logger.info(f"Tests per model: {args.num_tests}\n")

    passed_count = 0
    failed_models = []

    for model_path in model_paths:
        if validate_model(model_path, args.num_tests):
            passed_count += 1
        else:
            failed_models.append(model_path)

        logger.info("")  # Blank line between models

    # Final summary
    logger.info("=" * 60)
    logger.info("OVERALL VALIDATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total models validated: {len(model_paths)}")
    logger.info(f"Passed: {passed_count}")
    logger.info(f"Failed: {len(failed_models)}")

    if failed_models:
        logger.info("\nFailed models:")
        for model_path in failed_models:
            logger.info(f"  - {model_path}")
        logger.info("")
        sys.exit(1)
    else:
        logger.info("\n✅ All models validated successfully!")
        sys.exit(0)


if __name__ == '__main__':
    main()
