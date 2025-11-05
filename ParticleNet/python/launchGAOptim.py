#!/usr/bin/env python
"""
Genetic Algorithm optimization launcher for ParticleNet hyperparameter tuning.

Uses shared memory datasets to enable efficient parallel training of multiple
model configurations with minimal memory overhead.
"""

import os
import shutil
import logging
import argparse
import random
import json
import torch
import torch.multiprocessing as mp

from GAConfig import load_ga_config
from GATools import GeneticModule
import evaluateGAModels
from SharedDatasetManager import SharedDatasetManager
from DynamicDatasetLoader import DynamicDatasetLoader
from trainWorker import train_worker


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch GA hyperparameter optimization")
    parser.add_argument("--signal", required=True, type=str, help="Signal mass point (e.g., MHc130_MA100)")
    parser.add_argument("--channel", required=True, type=str, help="Channel (Run1E2Mu, Run3Mu, Combined)")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cpu or cuda:X)")
    parser.add_argument("--pilot", action="store_true", default=False, help="Use pilot datasets")
    parser.add_argument("--debug", action="store_true", default=False, help="Debug mode")
    return parser.parse_args()


def setup_output_directory(config, args):
    """Setup and validate output directory."""
    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    output_config = config.get_output_config()
    dataset_config = config.get_dataset_config()
    signal_full = dataset_config['signal_prefix'] + args.signal
    results_dir = output_config['results_dir']
    base_dir = f"{WORKDIR}/ParticleNet/{results_dir}/{args.channel}/multiclass/{signal_full}"

    if os.path.exists(base_dir):
        logging.info(f"Deleting existing directory: {base_dir}")
        shutil.rmtree(base_dir)

    return base_dir


def run_training_population(config, args, population, iteration, base_dir):
    """Launch parallel training using shared memory datasets."""

    logging.info("=" * 60)
    logging.info("Loading datasets into shared memory")
    logging.info("=" * 60)

    # Get configurations
    WORKDIR = os.environ.get("WORKDIR")
    dataset_config = config.get_dataset_config()
    train_params = config.get_training_parameters()
    bg_groups = config.get_background_groups()
    exec_config = config.get_execution_config()

    # Construct full sample names
    signal_full = dataset_config['signal_prefix'] + args.signal
    background_groups_full = {
        group_name: [dataset_config['background_prefix'] + sample for sample in samples]
        for group_name, samples in bg_groups.items()
    }

    # Initialize dataset manager and loader
    dataset_root = f"{WORKDIR}/ParticleNet/dataset"
    if dataset_config.get('use_bjets', False):
        dataset_root = dataset_root.replace('/dataset', '/dataset_bjets')

    loader = DynamicDatasetLoader(
        dataset_root=dataset_root,
        separate_bjets=dataset_config.get('use_bjets', False)
    )

    manager = SharedDatasetManager()

    # Load datasets into shared memory (ONE TIME ONLY!)
    train_data_list, valid_data_list = manager.prepare_shared_datasets(
        loader=loader,
        signal_sample=signal_full,
        background_groups=background_groups_full,
        channel=args.channel,
        train_folds=train_params['train_folds'],
        valid_folds=train_params['valid_folds'],
        pilot=args.pilot,
        max_events_per_fold=train_params.get('max_events_per_fold_per_class'),
        balance_weights=train_params.get('balance_weights', True),
        random_state=42
    )

    # Display memory usage statistics
    memory_stats = manager.get_memory_usage()
    logging.info(f"Shared memory usage: {memory_stats['total_gb']:.2f} GB")

    # Prepare arguments for workers
    args_dict = {
        'signal': args.signal,
        'channel': args.channel,
        'device': args.device,
        'iteration': iteration,
        'pilot': args.pilot,
        'debug': args.debug
    }

    # Extract hyperparameters for all models
    population_hyperparams = []
    for model_config in population:
        nNodes, optimizer, initLR, weight_decay, scheduler = model_config["chromosome"]
        hyperparams = {
            'nNodes': nNodes,
            'optimizer': optimizer,
            'initLR': initLR,
            'weight_decay': weight_decay,
            'scheduler': scheduler
        }
        population_hyperparams.append(hyperparams)

    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # Already set, that's fine
        pass

    logging.info(f"[Iter {iteration}] Spawning {len(population)} worker processes...")

    # Launch workers using multiprocessing.spawn
    # Note: spawn automatically handles process creation and joining
    try:
        mp.spawn(
            train_worker,
            args=(population_hyperparams, train_data_list, valid_data_list, args_dict, config),
            nprocs=len(population),
            join=True  # Wait for all processes to complete
        )
        logging.info(f"[Iter {iteration}] All {len(population)} workers completed")
    except Exception as e:
        logging.error(f"Error during parallel training: {e}")
        raise

    # Clear cache after training to free memory
    manager.clear_cache()


def run_overfitting_check(config, args, ga_module, iteration, base_dir):
    """Run overfitting detection and filter population."""
    logging.info("=" * 60)
    logging.info("OVERFITTING DETECTION & FILTERING")
    logging.info("=" * 60)

    valid_indices = evaluateGAModels.evaluate_iteration(
        config, args.signal, args.channel, args.device, iteration, base_dir, pilot=args.pilot
    )

    original_size = len(ga_module.population)
    ga_module.population = [ga_module.population[i] for i in valid_indices]
    filtered_count = original_size - len(ga_module.population)

    logging.info(f"Filtered: {filtered_count}/{original_size} overfitted models")
    logging.info(f"Valid (survivors): {len(ga_module.population)} models")
    logging.info("=" * 60)


def update_population_fitness(config, ga_module, args, iteration, base_dir):
    """Update fitness values and save population summary."""
    if len(ga_module.population) == 0:
        logging.warning("Population is empty - skipping fitness update")
        return

    output_config = config.get_output_config()
    fitness_metric = config.get_fitness_metric()
    ga_params = config.get_ga_parameters()
    penalty_weight = ga_params.get('overfitting_penalty_weight', 0.0)

    ga_subdir = output_config['ga_subdir_pattern'].format(iteration=iteration)
    json_dir = f"{base_dir}/{ga_subdir}/{output_config['json_subdir']}"

    if penalty_weight > 0:
        logging.info(f"Using overfitting penalty: λ = {penalty_weight}")

    ga_module.updatePopulation(
        args.channel,
        metric=fitness_metric,
        read_from=json_dir,
        model_name_pattern=output_config['model_name_pattern'],
        penalty_weight=penalty_weight
    )

    # Log results
    best = ga_module.bestChromosome()
    mean_fitness = ga_module.meanFitness()
    logging.info(f"Best chromosome: {best}")
    logging.info(f"Best fitness: {best['fitness']:.6f}")
    logging.info(f"Mean fitness: {mean_fitness:.6f}")

    # Save population summary
    population_json = f"{json_dir}/{output_config['population_summary_name']}"
    os.makedirs(os.path.dirname(population_json), exist_ok=True)
    ga_module.savePopulation(population_json)
    logging.info(f"Population saved to: {population_json}")


def regenerate_models(ga_module, count):
    """Generate random models from the pool."""
    logging.info(f"Generating {count} random models...")
    return [{'chromosome': random.choice(ga_module.pool), 'fitness': None} for _ in range(count)]


def main():
    """Main GA optimization workflow."""
    args = parse_arguments()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    if args.pilot:
        logging.info("=" * 60)
        logging.info("PILOT MODE ENABLED - Using pilot datasets")
        logging.info("=" * 60)

    # Load configuration and setup
    logging.info("Loading GA configuration...")
    config = load_ga_config()
    print("\n" + config.summary())

    base_dir = setup_output_directory(config, args)

    # Get GA parameters
    ga_params = config.get_ga_parameters()
    population_size = ga_params['population_size']
    max_iterations = ga_params['max_iterations']
    evolution_ratio = ga_params['evolution_ratio']
    mutation_thresholds = ga_params['mutation_thresholds']
    overfitting_config = config.get_overfitting_config()

    # Initialize GeneticModule
    logging.info("Initializing genetic algorithm module...")
    ga_module = GeneticModule()

    for space in config.generate_hyperparameter_spaces():
        ga_module.setConfigSpace(space)

    ga_module.generatePopulation()
    ga_module.randomGeneration(n_population=population_size)
    logging.info(f"Generated initial population of {population_size} configurations")

    # ========== Generation 0 ==========
    logging.info("=" * 60)
    logging.info("GENERATION 0 - Initial Population")
    logging.info("=" * 60)

    run_training_population(config, args, ga_module.population, 0, base_dir)

    if overfitting_config['enabled']:
        run_overfitting_check(config, args, ga_module, 0, base_dir)

    update_population_fitness(config, ga_module, args, 0, base_dir)

    # ========== Evolution Loop ==========
    for iteration in range(1, max_iterations):
        logging.info("=" * 60)
        logging.info(f"GENERATION {iteration}")
        logging.info("=" * 60)

        if len(ga_module.population) == 0:
            logging.warning("Population extinct! Regenerating models...")
            ga_module.population = regenerate_models(ga_module, population_size)
        elif len(ga_module.population) < int(population_size * 0.25):
            logging.warning(f"Population critically low ({len(ga_module.population)}). Regenerating...")
            needed = population_size - len(ga_module.population)
            ga_module.population.extend(regenerate_models(ga_module, needed))
        else:
            # Normal evolution
            # evolution_ratio = fraction of NEW children to create
            num_children = int(len(ga_module.population) * evolution_ratio)
            num_survivors = len(ga_module.population) - num_children

            logging.info(f"Population: {len(ga_module.population)}")
            logging.info(f"Survivors: {num_survivors}, New children: {num_children} (evolution_ratio: {evolution_ratio:.2f})")

            ga_module.evolution(mutation_thresholds, evolution_ratio)

        # Train the new generation
        run_training_population(config, args, ga_module.population, iteration, base_dir)

        if overfitting_config['enabled']:
            run_overfitting_check(config, args, ga_module, iteration, base_dir)

        update_population_fitness(config, ga_module, args, iteration, base_dir)

    # Final results
    best = ga_module.bestChromosome()
    logging.info("=" * 60)
    logging.info("GA OPTIMIZATION COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Best configuration found: {best}")
    logging.info(f"Best fitness: {best['fitness']:.6f}")

    # Save final results
    final_results = {
        'signal': args.signal,
        'channel': args.channel,
        'best_chromosome': best,
        'configuration': config.to_dict()
    }
    final_path = f"{base_dir}/ga_optimization_results.json"
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    logging.info(f"Final results saved to: {final_path}")


if __name__ == "__main__":
    main()