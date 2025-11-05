#!/usr/bin/env python
"""Genetic Algorithm optimization launcher for ParticleNet multi-class training."""

import os
import shutil
import logging
import argparse
import subprocess
import random
import json
from time import sleep

from GAConfig import load_ga_config
from GATools import GeneticModule
import evaluateGAModels


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
    """Launch parallel training for entire population."""
    exec_config = config.get_execution_config()
    processes = []

    for idx in range(len(population)):
        nNodes, optimizer, initLR, weight_decay, scheduler = population[idx]["chromosome"]

        command = f"python/trainMultiClassForGA.py"
        command += f" --signal {args.signal} --channel {args.channel} --iter {iteration} --idx {idx}"
        command += f" --nNodes {nNodes} --optimizer {optimizer} --initLR {initLR}"
        command += f" --weight_decay {weight_decay} --scheduler {scheduler} --device {args.device}"
        if args.pilot:
            command += " --pilot"
        if args.debug:
            command += " --debug"

        logging.info(f"[Iter {iteration}] Starting model {idx}: nNodes={nNodes}, opt={optimizer}")

        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((idx, proc))
        sleep(exec_config['process_delay_seconds'])

    # Wait for all processes to complete
    for idx, proc in processes:
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            logging.error(f"Model {idx} failed with return code {proc.returncode}")
            logging.error(f"stderr: {stderr.decode()}")
            raise RuntimeError(f"Training failed for model {idx}")

    logging.info(f"[Iter {iteration}] All {len(population)} models completed")


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
        logging.info(f"GENERATION {iteration} - Preparation")
        logging.info("=" * 60)

        num_survivors = len(ga_module.population)
        logging.info(f"Starting with {num_survivors} survivors from previous iteration")

        if num_survivors > 0:
            logging.info(f"Evolving from {num_survivors} survivors (ratio={evolution_ratio})")
            ga_module.evolution(thresholds=mutation_thresholds, ratio=evolution_ratio)
        else:
            logging.info("No survivors - will regenerate entire population")
            ga_module.population = []

        # Regenerate to fill population
        needed = population_size - len(ga_module.population)
        if needed > 0:
            logging.info(f"Regenerating {needed} models to reach population size {population_size}")
            ga_module.population.extend(regenerate_models(ga_module, needed))
        elif needed < 0:
            raise RuntimeError(f"Population size ({len(ga_module.population)}) exceeds target ({population_size})")

        logging.info(f"Final population ready: {len(ga_module.population)} models")

        # Train, evaluate, and update
        logging.info("=" * 60)
        logging.info(f"GENERATION {iteration} - Training")
        logging.info("=" * 60)

        run_training_population(config, args, ga_module.population, iteration, base_dir)

        if overfitting_config['enabled']:
            run_overfitting_check(config, args, ga_module, iteration, base_dir)

        update_population_fitness(config, ga_module, args, iteration, base_dir)

    # ========== Final Summary ==========
    logging.info("=" * 60)
    logging.info("GA OPTIMIZATION COMPLETED")
    logging.info("=" * 60)
    if len(ga_module.population) > 0:
        final_best = ga_module.bestChromosome()
        logging.info(f"Final best chromosome: {final_best['chromosome']}")
        logging.info(f"Final best fitness: {final_best['fitness']:.6f}")
        logging.info(f"Model: {final_best.get('model', 'N/A')}")
    else:
        logging.warning("No valid models found - all models were filtered as overfitted")
    logging.info(f"Results directory: {base_dir}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
