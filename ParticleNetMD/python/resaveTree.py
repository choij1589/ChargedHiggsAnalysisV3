#!/usr/bin/env python
"""
Re-save ROOT prediction trees with channel_id and run_id branches.

Reuses existing infrastructure to re-run only the tree-saving step
without retraining. Useful for adding new branches to existing trees.

Single-model mode:
    python python/resaveTree.py --signal MHc130_MA90 --channel Combined \
        --model-path LambdaSweep/.../models/discoL0p1...pt \
        --json-path LambdaSweep/.../discoL0p1...json \
        --tree-path LambdaSweep/.../trees/discoL0p1...root

Batch mode (all models under a directory):
    python python/resaveTree.py --signal MHc130_MA90 --channel Combined \
        --base-dir LambdaSweep/Combined/MHc130_MA90/fold-4
"""

import argparse
import glob
import json
import logging
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

import torch

from DataPipeline import create_data_pipeline
from MultiClassModels import create_multiclass_model
from ResultPersistence import create_result_persistence


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Re-save ROOT prediction trees with channel_id/run_id branches"
    )
    parser.add_argument("--signal", required=True, type=str,
                        help="Signal name without prefix (e.g., MHc130_MA90)")
    parser.add_argument("--channel", required=True, type=str,
                        choices=["Run1E2Mu", "Run3Mu", "Combined"],
                        help="Channel")

    # Single-model mode
    parser.add_argument("--model-path", default=None, type=str,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--json-path", default=None, type=str,
                        help="Path to GA-compatible JSON with hyperparameters")
    parser.add_argument("--tree-path", default=None, type=str,
                        help="Output path for ROOT tree (.root)")

    # Batch mode
    parser.add_argument("--base-dir", default=None, type=str,
                        help="Base directory containing models/ and trees/ subdirs. "
                             "Auto-discovers all .pt models and re-saves their trees.")

    parser.add_argument("--config", default=None, type=str,
                        help="Path to SglConfig JSON (default: configs/SglConfig.json)")
    parser.add_argument("--pilot", action="store_true", default=False,
                        help="Use pilot (small) datasets")
    parser.add_argument("--device", default="cpu", type=str,
                        help="Device for inference (default: cpu)")

    args = parser.parse_args()

    # Validate: either single-model or batch mode
    single = args.model_path is not None
    batch = args.base_dir is not None
    if not single and not batch:
        parser.error("Provide either --model-path/--json-path/--tree-path (single) or --base-dir (batch)")
    if single and batch:
        parser.error("Cannot use both --model-path and --base-dir")
    if single and (args.json_path is None or args.tree_path is None):
        parser.error("Single mode requires --model-path, --json-path, and --tree-path")

    return args


def discover_models(base_dir):
    """Discover model/json/tree triplets under base_dir.

    Expects layout:
        base_dir/models/<name>.pt
        base_dir/<name>.json
        base_dir/trees/<name>.root
    """
    model_dir = os.path.join(base_dir, "models")
    tree_dir = os.path.join(base_dir, "trees")

    pt_files = sorted(glob.glob(os.path.join(model_dir, "*.pt")))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {model_dir}")

    triplets = []
    for pt_path in pt_files:
        name = os.path.splitext(os.path.basename(pt_path))[0]
        json_path = os.path.join(base_dir, name + ".json")
        tree_path = os.path.join(tree_dir, name + ".root")

        if not os.path.exists(json_path):
            logging.warning(f"Skipping {name}: JSON not found at {json_path}")
            continue

        triplets.append((pt_path, json_path, tree_path))

    return triplets


def _infer_graph_features_from_checkpoint(checkpoint):
    """Infer num_graph_features from checkpoint dense1 weight shape.

    dense1.weight shape is (nNodes, 3*nNodes + num_graph_features).
    """
    state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
    if 'dense1.weight' in state:
        total = state['dense1.weight'].shape[1]
        # bn0 input = 3 * nNodes + num_graph_features
        nNodes = state['dense1.weight'].shape[0]
        return total - 3 * nNodes
    return None


def load_model(model_path, json_path, device, config):
    """Load a model checkpoint using hyperparameters from its JSON."""
    with open(json_path, 'r') as f:
        hyperparams = json.load(f)['hyperparameters']

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Infer num_graph_features from checkpoint to handle stale JSON values
    num_graph_features = _infer_graph_features_from_checkpoint(checkpoint)
    if num_graph_features is None:
        num_graph_features = hyperparams.get('num_graph_features', 8)

    json_ngf = hyperparams.get('num_graph_features')
    if json_ngf is not None and json_ngf != num_graph_features:
        logging.warning(f"JSON says num_graph_features={json_ngf}, "
                        f"checkpoint has {num_graph_features}. Using checkpoint value.")

    model = create_multiclass_model(
        model_type=hyperparams.get('model_type', config.args.model),
        num_node_features=hyperparams.get('num_node_features', 9),
        num_graph_features=num_graph_features,
        num_classes=hyperparams.get('num_classes', config.num_classes),
        num_hidden=hyperparams.get('num_hidden', config.args.nNodes),
        dropout_p=hyperparams.get('dropout_p', config.args.dropout_p)
    ).to(device)

    model_state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(model_state)
    model.eval()
    return model


def main():
    args = parse_arguments()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, script_dir)
    from trainMultiClass import Config

    # Build config (dataset is the same for all models of a given signal+channel)
    config = Config(args.signal, args.channel, args.config)
    config.args.pilot = args.pilot
    config.args.device = args.device

    # Determine model triplets
    if args.base_dir:
        triplets = discover_models(args.base_dir)
        logging.info(f"Batch mode: found {len(triplets)} models in {args.base_dir}")
    else:
        triplets = [(args.model_path, args.json_path, args.tree_path)]

    # Load dataset ONCE
    logging.info(f"Loading dataset for {args.signal} / {args.channel}...")
    data_pipeline = create_data_pipeline(config)
    data_pipeline.create_datasets()
    data_pipeline.create_data_loaders(batch_size=config.args.batch_size)
    logging.info("Dataset ready.")

    device = torch.device(args.device)
    persistence = create_result_persistence(config)

    # Loop over models
    for i, (model_path, json_path, tree_path) in enumerate(triplets):
        name = os.path.splitext(os.path.basename(model_path))[0]
        logging.info(f"[{i+1}/{len(triplets)}] {name}")

        model = load_model(model_path, json_path, device, config)
        persistence.save_predictions_to_root(model, data_pipeline, device, tree_path)
        logging.info(f"  -> {tree_path}")

    logging.info(f"Done. Re-saved {len(triplets)} tree(s).")


if __name__ == "__main__":
    main()
