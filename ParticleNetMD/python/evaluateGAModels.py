#!/usr/bin/env python
"""
Batch Overfitting Evaluation for GA Iteration in ParticleNetMD.

Evaluates all models in a GA iteration using 16 KS tests for overfitting detection.
"""

import os
import json
import logging
import subprocess
import sys
from time import sleep
from OverfittingDetector import OverfittingDetector


def evaluate_single_model_standalone(signal, channel, device, iteration, idx, pilot=False):
    """Standalone entry point for evaluating a single model via subprocess."""
    from GAConfig import load_ga_config

    config = load_ga_config()
    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    output_config = config.get_output_config()
    dataset_config = config.get_dataset_config()
    signal_full = dataset_config['signal_prefix'] + signal
    results_dir = output_config['results_dir']

    # ParticleNetMD-specific path
    base_dir = f"{WORKDIR}/ParticleNetMD/{results_dir}/{channel}/multiclass/{signal_full}"

    detector = OverfittingDetector(config, signal, channel, device, pilot=pilot)
    ga_subdir = output_config['ga_subdir_pattern'].format(iteration=iteration)
    output_dir = os.path.join(base_dir, ga_subdir, "overfitting_diagnostics")
    os.makedirs(output_dir, exist_ok=True)

    model_name = output_config['model_name_pattern'].format(idx=idx)
    model_path = os.path.join(base_dir, ga_subdir, output_config['models_subdir'], f"{model_name}.pt")
    result_file = os.path.join(output_dir, f"model{idx}_result.json")

    result = _evaluate_model_safe(detector, model_path, iteration, idx, output_dir)

    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    logging.info(f"Model {idx} evaluation complete. Result saved to {result_file}")


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


def _run_parallel_evaluation(config, signal, channel, device, iteration, base_dir, population_size, start_idx, pilot):
    """Run evaluations in parallel using subprocess."""
    output_config = config.get_output_config()
    ga_subdir = output_config['ga_subdir_pattern'].format(iteration=iteration)
    output_dir = os.path.join(base_dir, ga_subdir, "overfitting_diagnostics")

    logging.info(f"Launching {population_size} parallel subprocess evaluations")

    processes = []
    for local_idx in range(population_size):
        global_idx = start_idx + local_idx
        command = f"python/evaluateGAModels.py --single-model --signal {signal} --channel {channel}"
        command += f" --device {device} --iteration {iteration} --idx {global_idx}"
        if pilot:
            command += " --pilot"

        logging.info(f"Launching evaluation for model {global_idx}")
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((global_idx, proc))
        sleep(0.1)

    # Collect results
    results = []
    for global_idx, proc in processes:
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            logging.error(f"Model {global_idx} evaluation failed with return code {proc.returncode}")
            logging.error(f"stderr: {stderr.decode()}")
            results.append({
                'model_idx': global_idx, 'is_overfitted': True, 'kolmogorov_results': None,
                'error': f"Subprocess failed with code {proc.returncode}"
            })
        else:
            result_file = os.path.join(output_dir, f"model{global_idx}_result.json")
            if os.path.exists(result_file):
                with open(result_file, 'r') as f:
                    results.append(json.load(f))
            else:
                logging.error(f"Result file not found for model {global_idx}: {result_file}")
                results.append({
                    'model_idx': global_idx, 'is_overfitted': True,
                    'kolmogorov_results': None, 'error': "Result file not found"
                })

    logging.info(f"All {population_size} model evaluations completed")
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
        results = _run_parallel_evaluation(
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
    parser.add_argument("--single-model", action="store_true", default=False,
                        help="Evaluate a single model (for subprocess parallel execution)")
    parser.add_argument("--signal", required=True, type=str, help="Signal mass point (e.g., MHc130_MA100)")
    parser.add_argument("--channel", required=True, type=str, help="Channel (Run1E2Mu, Run3Mu, Combined)")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cpu or cuda:X)")
    parser.add_argument("--iteration", required=True, type=int, help="GA iteration number")
    parser.add_argument("--idx", type=int, help="Model index (required for --single-model)")
    parser.add_argument("--pilot", action="store_true", default=False, help="Use pilot datasets")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.single_model:
        if args.idx is None:
            raise ValueError("--idx is required when using --single-model")

        evaluate_single_model_standalone(
            signal=args.signal, channel=args.channel, device=args.device,
            iteration=args.iteration, idx=args.idx, pilot=args.pilot
        )
    else:
        config = load_ga_config()
        WORKDIR = os.environ.get("WORKDIR")
        if not WORKDIR:
            raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

        output_config = config.get_output_config()
        dataset_config = config.get_dataset_config()
        signal_full = dataset_config['signal_prefix'] + args.signal
        results_dir = output_config['results_dir']

        # ParticleNetMD-specific path
        base_dir = f"{WORKDIR}/ParticleNetMD/{results_dir}/{args.channel}/multiclass/{signal_full}"

        overfitted_indices = evaluate_iteration(
            config, args.signal, args.channel, args.device,
            args.iteration, base_dir, pilot=args.pilot
        )

        print(f"\nOverfitted model indices: {overfitted_indices}")
