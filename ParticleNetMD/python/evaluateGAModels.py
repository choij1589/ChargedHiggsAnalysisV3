#!/usr/bin/env python
"""
Batch Overfitting Evaluation for GA Iteration in ParticleNetMD.

Evaluates all models in a GA iteration using 16 KS tests for overfitting detection.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import json
import logging
import torch
import numpy as np
from OverfittingDetector import OverfittingDetector


def _evaluate_model_safe(detector, model_path, iteration, idx, output_dir):
    """Safely evaluate a model with error handling."""
    result = {'model_idx': idx}

    if not os.path.exists(model_path):
        result['is_overfitted'] = True
        result['kolmogorov_results'] = None
        result['error'] = f"Model checkpoint not found: {model_path}"
    else:
        try:
            is_overfitted, ks_results = detector.check_overfitting(model_path, iteration, idx, output_dir)
            result['is_overfitted'] = is_overfitted
            result['kolmogorov_results'] = ks_results
        except Exception as e:
            logging.error(f"Error evaluating model {idx}: {e}")
            result['is_overfitted'] = True
            result['kolmogorov_results'] = None
            result['error'] = str(e)

    return result


def _make_shared_batch(data_list):
    """Collate List[Data] into a Batch with all tensors in OS shared memory."""
    from torch_geometric.data import Batch
    batch = Batch.from_data_list(data_list)
    for key in batch.keys():
        v = getattr(batch, key)
        if torch.is_tensor(v):
            v.share_memory_()
    return batch


def _load_eval_shared_datasets(config, signal_full, channel, bg_groups, WORKDIR, pilot):
    """Load and share train + test datasets. Pilot caps: 8000 events/fold/class."""
    from DynamicDatasetLoader import DynamicDatasetLoader
    dataset_root = f"{WORKDIR}/ParticleNetMD/dataset"
    loader = DynamicDatasetLoader(dataset_root=dataset_root)
    train_params = config.get_training_parameters()
    overfitting_config = config.get_overfitting_config()
    dataset_config = config.get_dataset_config()

    # Apply background prefix (same as launchGAOptim.py)
    bg_prefix = dataset_config['background_prefix']
    bg_groups_full = {
        group: [bg_prefix + s for s in samples]
        for group, samples in bg_groups.items()
    }

    max_events = 8000 if pilot else train_params.get('max_events_per_fold_per_class')
    train_folds = [train_params['train_folds'][0]] if pilot else train_params['train_folds']
    test_folds = overfitting_config.get('test_folds', [4])

    train_data = loader.load_multiclass_with_subsampling(
        signal_sample=signal_full, background_groups=bg_groups_full, channel=channel,
        fold_list=train_folds, max_events_per_fold=max_events,
        balance_weights=train_params.get('balance_weights', True)
    )
    test_data = loader.load_multiclass_with_subsampling(
        signal_sample=signal_full, background_groups=bg_groups_full, channel=channel,
        fold_list=test_folds, max_events_per_fold=max_events,
        balance_weights=train_params.get('balance_weights', True)
    )
    logging.info(f"Shared datasets: {len(train_data)} train, {len(test_data)} test events")
    return _make_shared_batch(train_data), _make_shared_batch(test_data)


def evaluate_spawn_worker(worker_rank, model_paths, output_dir, device_str,
                          shared_train_batch, shared_test_batch,
                          model_config, train_params, p_threshold,
                          bin_merge_threshold, bin_merge_max_iterations):
    """Worker function for mp.spawn: one model per worker."""
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
    import torch
    import json
    import numpy as np
    from SharedWorkerUtils import make_dataloader_from_batch
    from MultiClassModels import create_multiclass_model
    from OverfittingDetector import (perform_ks_tests_comprehensive, save_ks_results)

    idx = worker_rank
    model_path = model_paths[idx]
    result_file = os.path.join(output_dir, f"model{idx}_result.json")
    result = {'model_idx': idx}

    try:
        device = torch.device(device_str)
        checkpoint = torch.load(model_path, weights_only=True)
        nNodes = checkpoint['model_state_dict']['conv1.mlp.0.weight'].shape[0]
        model = create_multiclass_model(
            model_type=model_config['default_model'],
            num_node_features=9, num_graph_features=8,
            num_classes=4, num_hidden=nNodes,
            dropout_p=train_params['dropout_p']
        ).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        def get_preds(shared_batch):
            loader = make_dataloader_from_batch(shared_batch, batch_size=2048, shuffle=False)
            labels, scores, weights = [], [], []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)
                    probs = torch.softmax(logits, dim=1)
                    labels.append(batch.y.cpu().numpy())
                    scores.append(probs.cpu().numpy())
                    weights.append(batch.weight.cpu().numpy())
            return np.concatenate(labels), np.concatenate(scores), np.concatenate(weights)

        train_labels, train_scores, train_weights = get_preds(shared_train_batch)
        test_labels, test_scores, test_weights = get_preds(shared_test_batch)

        model_output_dir = os.path.join(output_dir, f"model{idx}")
        os.makedirs(model_output_dir, exist_ok=True)
        ks_results, histograms = perform_ks_tests_comprehensive(
            train_labels, train_scores, train_weights,
            test_labels, test_scores, test_weights,
            p_threshold=p_threshold,
            bin_merge_threshold=bin_merge_threshold,
            bin_merge_max_iterations=bin_merge_max_iterations
        )
        save_ks_results(ks_results, histograms, model_output_dir)
        result['is_overfitted'] = any(r['is_overfitted'] for r in ks_results.values())
        result['kolmogorov_results'] = ks_results

        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    except Exception as e:
        import logging as _logging
        _logging.error(f"evaluate_spawn_worker model {idx}: {e}")
        result['is_overfitted'] = True
        result['error'] = str(e)

    os.makedirs(output_dir, exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)


def _run_spawn_evaluation(config, signal, channel, device, iteration,
                          base_dir, population_size, start_idx, pilot):
    """Load data once, share via mp.spawn across all model evaluators."""
    import torch.multiprocessing as mp
    from SharedWorkerUtils import setup_spawn_method

    setup_spawn_method()

    WORKDIR = os.environ.get("WORKDIR")
    dataset_config = config.get_dataset_config()
    signal_full = dataset_config['signal_prefix'] + signal
    bg_groups = config.get_background_groups()

    shared_train, shared_test = _load_eval_shared_datasets(
        config, signal_full, channel, bg_groups, WORKDIR, pilot
    )

    output_config = config.get_output_config()
    ga_subdir = output_config['ga_subdir_pattern'].format(iteration=iteration)
    output_dir = os.path.join(base_dir, ga_subdir, "overfitting_diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    model_config = config.get_model_config()
    train_params = config.get_training_parameters()
    overfitting_config = config.get_overfitting_config()

    model_paths = [
        os.path.join(base_dir, ga_subdir, output_config['models_subdir'],
                     f"{output_config['model_name_pattern'].format(idx=start_idx+i)}.pt")
        for i in range(population_size)
    ]

    try:
        mp.spawn(
            evaluate_spawn_worker,
            args=(model_paths, output_dir, device, shared_train, shared_test,
                  model_config, train_params,
                  overfitting_config.get('p_value_threshold', 0.05),
                  overfitting_config.get('bin_merge_threshold', 1e-6),
                  overfitting_config.get('bin_merge_max_iterations', 100)),
            nprocs=population_size,
            join=True
        )
    except Exception as e:
        logging.warning(f"mp.spawn finished with errors: {e}; collecting partial results")

    results = []
    for i in range(population_size):
        global_idx = start_idx + i
        result_file = os.path.join(output_dir, f"model{global_idx}_result.json")
        if os.path.exists(result_file):
            with open(result_file) as f:
                results.append(json.load(f))
        else:
            results.append({'model_idx': global_idx, 'is_overfitted': True,
                            'error': 'result file not written'})
    return results


def _run_sequential_evaluation(config, base_dir, signal, channel, device, iteration, population_size, start_idx, pilot):
    """Run evaluations sequentially."""
    output_config = config.get_output_config()
    ga_subdir = output_config['ga_subdir_pattern'].format(iteration=iteration)
    detector = OverfittingDetector(config, signal, channel, device, pilot=pilot)
    output_dir = os.path.join(base_dir, ga_subdir, "overfitting_diagnostics")
    results = []

    for local_idx in range(population_size):
        global_idx = start_idx + local_idx
        model_name = output_config['model_name_pattern'].format(idx=global_idx)
        model_path = os.path.join(base_dir, ga_subdir, output_config['models_subdir'], f"{model_name}.pt")

        if not os.path.exists(model_path):
            logging.warning(f"Model {global_idx} checkpoint not found: {model_path}")
            continue

        result = _evaluate_model_safe(detector, model_path, iteration, global_idx, output_dir)
        results.append(result)

    return results


def evaluate_iteration(config, signal, channel, device, iteration, base_dir, pilot=False, parallel=True, start_idx=0, population_size=None):
    """Evaluate all models in a GA iteration for overfitting.

    Args:
        population_size: Number of models to evaluate. If None, uses config value.
                        Should match the actual number of trained models.
    """
    logging.info("=" * 60)
    logging.info(f"OVERFITTING EVALUATION - Iteration {iteration}")
    logging.info("=" * 60)

    ga_params = config.get_ga_parameters()
    output_config = config.get_output_config()

    # Use provided population_size or fall back to config
    if population_size is None:
        population_size = ga_params['population_size']

    ga_subdir = output_config['ga_subdir_pattern'].format(iteration=iteration)
    output_dir = os.path.join(base_dir, ga_subdir, "overfitting_diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Evaluating {population_size} models (indices {start_idx}-{start_idx+population_size-1})...")
    logging.info(f"Diagnostics will be saved to: {output_dir}")
    logging.info(f"Parallel mode: {parallel}")

    # Run evaluations
    if parallel:
        results = _run_spawn_evaluation(
            config, signal, channel, device, iteration, base_dir, population_size, start_idx, pilot
        )
    else:
        results = _run_sequential_evaluation(
            config, base_dir, signal, channel, device, iteration, population_size, start_idx, pilot
        )

    # Save summary and print statistics
    save_evaluation_summary(results, iteration, output_dir)
    overfitted_indices = [r['model_idx'] for r in results if r['is_overfitted']]

    total, overfitted = len(results), len(overfitted_indices)
    valid = total - overfitted
    logging.info("=" * 60)
    logging.info("OVERFITTING EVALUATION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Total models evaluated: {total}")
    logging.info(f"Valid models (not overfitted): {valid}")
    logging.info(f"Overfitted models: {overfitted}")

    if overfitted > 0:
        logging.info(f"Overfitted model indices: {overfitted_indices}")

    logging.info("=" * 60)
    return overfitted_indices


def _compute_summary_statistics(results):
    """Compute summary statistics from evaluation results."""
    total_models = len(results)
    valid_models = sum(1 for r in results if not r['is_overfitted'])
    overfitted_models = total_models - valid_models
    overfitted_indices = [r['model_idx'] for r in results if r['is_overfitted']]

    model_details = []
    for r in results:
        model_summary = {'model_idx': r['model_idx'], 'is_overfitted': r['is_overfitted']}

        if r.get('kolmogorov_results'):
            ks_results = r['kolmogorov_results']
            model_summary['per_test_p_values'] = {
                test_name: res['p_value'] for test_name, res in ks_results.items()
            }
            model_summary['failed_tests'] = [
                test_name for test_name, res in ks_results.items() if res['is_overfitted']
            ]
            model_summary['overall_min_p_value'] = min(
                (res['p_value'] for res in ks_results.values() if res['p_value'] is not None),
                default=None
            )
            model_summary['num_failed_tests'] = len(model_summary['failed_tests'])

        if 'error' in r:
            model_summary['error'] = r['error']

        model_details.append(model_summary)

    return {
        'total_models': total_models, 'valid_models': valid_models,
        'overfitted_models': overfitted_models, 'overfitted_model_indices': overfitted_indices,
        'model_details': model_details
    }


def _save_summary_json(summary, iteration, output_dir):
    """Save evaluation summary to JSON."""
    summary['iteration'] = iteration
    output_file = os.path.join(output_dir, 'overfitting_summary.json')
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Evaluation summary saved to: {output_file}")


def _save_summary_text(summary, iteration, output_dir):
    """Save evaluation summary to text file."""
    text_file = os.path.join(output_dir, 'overfitting_summary.txt')
    with open(text_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"OVERFITTING EVALUATION SUMMARY - Iteration {iteration}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total models evaluated: {summary['total_models']}\n")
        f.write(f"Valid models (not overfitted): {summary['valid_models']}\n")
        f.write(f"Overfitted models: {summary['overfitted_models']}\n\n")

        if summary['overfitted_models'] > 0:
            f.write(f"Overfitted model indices: {summary['overfitted_model_indices']}\n\n")

        f.write("=" * 60 + "\n")
        f.write("PER-MODEL RESULTS (16 KS Tests per Model)\n")
        f.write("=" * 60 + "\n\n")

        for model in summary['model_details']:
            idx = model['model_idx']
            status = "OVERFITTED" if model['is_overfitted'] else "VALID"
            f.write(f"Model {idx}: {status}\n")

            if 'overall_min_p_value' in model and model['overall_min_p_value'] is not None:
                f.write(f"  Overall min p-value: {model['overall_min_p_value']:.4f}\n")

            if 'num_failed_tests' in model:
                f.write(f"  Failed tests: {model['num_failed_tests']}/16\n")

            if 'failed_tests' in model and model['failed_tests']:
                f.write(f"  Failed test names: {', '.join(model['failed_tests'][:5])}")
                if len(model['failed_tests']) > 5:
                    f.write(f" ... (+{len(model['failed_tests'])-5} more)")
                f.write("\n")

            if 'error' in model:
                f.write(f"  Error: {model['error']}\n")

            f.write("\n")

    logging.info(f"Text summary saved to: {text_file}")


def save_evaluation_summary(results, iteration, output_dir):
    """Save overfitting evaluation summary to JSON and text."""
    summary = _compute_summary_statistics(results)
    _save_summary_json(summary, iteration, output_dir)
    _save_summary_text(summary, iteration, output_dir)
    return summary


if __name__ == "__main__":
    """Standalone execution for testing or manual evaluation."""
    import argparse
    from GAConfig import load_ga_config

    parser = argparse.ArgumentParser(description="Evaluate GA models for overfitting (ParticleNetMD)")
    parser.add_argument("--signal", required=True, type=str, help="Signal mass point (e.g., MHc130_MA100)")
    parser.add_argument("--channel", required=True, type=str, help="Channel (Run1E2Mu, Run3Mu, Combined)")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cpu or cuda:X)")
    parser.add_argument("--iteration", required=True, type=int, help="GA iteration number")
    parser.add_argument("--pilot", action="store_true", default=False, help="Use pilot datasets")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    config = load_ga_config()
    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    output_config = config.get_output_config()
    results_dir = output_config['results_dir']
    overfitting_config = config.get_overfitting_config()
    test_folds = overfitting_config.get('test_folds', [4])

    # ParticleNetMD-specific path: short signal name, fold/pilot at leaf level
    fold_dir = "pilot" if args.pilot else f"fold-{test_folds[0]}"
    base_dir = f"{WORKDIR}/ParticleNetMD/{results_dir}/{args.channel}/{args.signal}/{fold_dir}"

    overfitted_indices = evaluate_iteration(
        config, args.signal, args.channel, args.device,
        args.iteration, base_dir, pilot=args.pilot
    )

    print(f"\nOverfitted model indices: {overfitted_indices}")
