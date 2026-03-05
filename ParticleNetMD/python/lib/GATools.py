"""Genetic algorithm module for hyperparameter optimization.

Method adapted from: https://link.springer.com/content/pdf/10.1007/s11042-020-10139-6.pdf
"""

import os
import logging
import random
import numpy as np
import pandas as pd
from itertools import product


class GeneticModule:
    """Genetic algorithm implementation for hyperparameter search."""

    def __init__(self):
        self.config_space = []  # hyperparameter space
        self.pool = []          # all possible configurations
        self.population = []    # current population

    def setConfigSpace(self, genes):
        """Add gene values to configuration space and update pool."""
        self.config_space.append(genes)
        self.pool = list(product(*self.config_space))

    def generatePopulation(self, criteria=None):
        """Generate population from pool with optional filtering."""
        if criteria:
            self.pool = list(filter(criteria, self.pool))
        self.population = [
            {'chromosome': ch, 'fitness': None}
            for ch in random.sample(self.pool, len(self.pool))
        ]

    def randomGeneration(self, n_population):
        """Randomly sample n individuals from population."""
        self.population = random.choices(self.population, k=n_population)

    def updatePopulation(self, channel, metric="loss/valid", read_from=None, model_name_pattern="model{idx}", penalty_weight=0.0):
        """Update fitness values from JSON files with optional overfitting penalty.

        Args:
            channel: Channel name
            metric: Fitness metric format (e.g., "loss/valid")
            read_from: Directory containing model JSON files
            model_name_pattern: Pattern for model filenames
            penalty_weight: Weight lambda for overfitting penalty.
                          Fitness = valid_loss + lambda*(valid_loss - train_loss)
                          Set to 0 to disable penalty (default)
        """
        import json

        for idx, individual in enumerate(self.population):
            model_name = model_name_pattern.format(idx=idx)
            json_path = f"{read_from}/{model_name}.json"

            if not os.path.exists(json_path):
                raise FileNotFoundError(f"JSON file not found for model: {json_path}")

            with open(json_path, 'r') as f:
                data = json.load(f)

            individual['model'] = model_name

            # Extract fitness from epoch history (metric format: "loss/valid" or "acc/train")
            if '/' not in metric:
                raise ValueError(f"Invalid metric format '{metric}'. Expected 'metric_type/split_type'")

            metric_type, split_type = metric.split('/')
            history_key = f"{split_type}_{metric_type}"  # e.g., "valid_loss"

            if history_key not in data['epoch_history']:
                raise KeyError(
                    f"Metric '{history_key}' not found in epoch history. "
                    f"Available keys: {list(data['epoch_history'].keys())}"
                )

            # Get best (minimum) value from history
            history_values = data['epoch_history'][history_key]
            best_epoch_idx = int(np.argmin(history_values))
            base_fitness = float(history_values[best_epoch_idx])

            # Add overfitting penalty if enabled
            if penalty_weight > 0:
                # Get corresponding train metric
                train_history_key = f"train_{metric_type}"

                if train_history_key not in data['epoch_history']:
                    logging.warning(
                        f"Model {idx}: Train metric '{train_history_key}' not found. "
                        f"Skipping overfitting penalty."
                    )
                    penalty = 0.0
                else:
                    train_values = data['epoch_history'][train_history_key]

                    # Get train loss at the SAME epoch as best valid loss
                    best_train_at_best_epoch = float(train_values[best_epoch_idx])
                    gap = base_fitness - best_train_at_best_epoch  # valid_loss - train_loss at same epoch
                    penalty = penalty_weight * max(0, gap)  # Only penalize positive gap

                    logging.debug(
                        f"Model {idx}: best_epoch={best_epoch_idx}, "
                        f"valid={base_fitness:.4f}, train={best_train_at_best_epoch:.4f}, "
                        f"gap={gap:.4f}, penalty={penalty:.4f}"
                    )

                individual['fitness'] = base_fitness + penalty
            else:
                individual['fitness'] = base_fitness

    def rankSelection(self):
        """Select two parents using rank-based roulette wheel selection."""
        sorted_pop = sorted(self.population, key=lambda x: x['fitness'])
        ranks = list(range(1, len(sorted_pop) + 1))
        total_rank = sum(ranks)
        selection_probs = [rank / total_rank for rank in reversed(ranks)]

        # Select two different parents
        p1 = random.choices(sorted_pop, weights=selection_probs)[0]
        p2 = p1
        while p2 == p1:
            p2 = random.choices(sorted_pop, weights=selection_probs)[0]

        return [p1, p2]

    def uniformCrossover(self, parent1, parent2, _max_retries=1000):
        """Perform uniform crossover between two parents."""
        pool_set = set(self.pool)
        for _ in range(_max_retries):
            child_chromosome = tuple(
                random.choice(pair) for pair in zip(parent1['chromosome'], parent2['chromosome'])
            )
            if child_chromosome in pool_set:
                return {'chromosome': child_chromosome, 'fitness': None}
            parent1, parent2 = self.rankSelection()
        raise RuntimeError(f"uniformCrossover failed to produce a valid chromosome after {_max_retries} attempts")

    def displacementMutation(self, child, thresholds, _max_retries=1000):
        """Mutate child chromosome based on mutation thresholds."""
        pool_set = set(self.pool)
        for _ in range(_max_retries):
            mutated = tuple(
                random.choice(self.config_space[i]) if random.random() > thresh else gene
                for i, (gene, thresh) in enumerate(zip(child['chromosome'], thresholds))
            )
            if mutated in pool_set:
                return {'chromosome': mutated, 'fitness': None}
        raise RuntimeError(f"displacementMutation failed to produce a valid chromosome after {_max_retries} attempts")

    def evolution(self, thresholds, ratio):
        """Evolve population using selection, crossover, and mutation."""
        parents = sorted(self.population, key=lambda x: x['fitness'])
        n_birth = int(len(parents) * ratio)
        children = []

        while len(children) < n_birth:
            p1, p2 = self.rankSelection()
            child = self.uniformCrossover(p1, p2)
            mutation = self.displacementMutation(child, thresholds)
            if mutation['chromosome'] not in [c['chromosome'] for c in children]:
                children.append(mutation)

        self.population = parents[:len(self.population) - n_birth] + children

    def bestChromosome(self):
        """Return best individual from population."""
        if not self.population:
            raise ValueError("Cannot get best chromosome from empty population")
        return min(self.population, key=lambda x: x['fitness'])

    def meanFitness(self):
        """Return mean fitness of population."""
        if not self.population:
            raise ValueError("Cannot compute mean fitness from empty population")
        fitness_values = [x['fitness'] for x in self.population if x['fitness'] is not None]
        if not fitness_values:
            raise ValueError("No valid fitness values in population")
        return np.mean(fitness_values)

    def savePopulation(self, path):
        """Save population to CSV file."""
        df = pd.DataFrame([
            {'model': p['model'], 'chromosome': p['chromosome'], 'fitness': p['fitness']}
            for p in self.population
        ])
        df.to_csv(path, index=False)

    def loadPopulation(self, path):
        """Load population from a previously saved CSV file.

        Reconstructs self.population with chromosome tuples and fitness values.
        Injects loaded chromosomes into self.pool if missing (handles seed mismatch).

        Args:
            path: Path to model_info.csv saved by savePopulation()
        """
        import ast

        if not os.path.exists(path):
            raise FileNotFoundError(f"Population CSV not found: {path}")

        df = pd.read_csv(path)
        required_cols = {'model', 'chromosome', 'fitness'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV missing required columns. Expected {required_cols}, got {set(df.columns)}")

        self.population = []
        pool_set = set(self.pool)

        for _, row in df.iterrows():
            chromosome = ast.literal_eval(row['chromosome'])
            chromosome = tuple(chromosome)
            fitness = float(row['fitness'])

            # Safety: inject into pool if missing (handles seed mismatch)
            if chromosome not in pool_set:
                self.pool.append(chromosome)
                pool_set.add(chromosome)
                logging.warning(f"Injected missing chromosome into pool: {chromosome}")

            self.population.append({
                'model': row['model'],
                'chromosome': chromosome,
                'fitness': fitness
            })

        logging.info(f"Loaded population of {len(self.population)} from {path}")

    def filterPopulation(self, overfitted_indices):
        """Remove overfitted models from population.

        Args:
            overfitted_indices: List of model indices to remove

        Returns:
            Number of models removed
        """
        if not overfitted_indices:
            return 0

        original_size = len(self.population)
        overfitted_set = set(overfitted_indices)
        self.population = [
            individual for idx, individual in enumerate(self.population)
            if idx not in overfitted_set
        ]
        removed = original_size - len(self.population)

        logging.info(f"Filtered {removed} overfitted models from population")
        return removed

    def regeneratePopulation(self, target_size):
        """Regenerate population to target size by sampling from pool.

        Args:
            target_size: Desired population size

        Returns:
            Number of new individuals added
        """
        current_size = len(self.population)
        if current_size >= target_size:
            return 0

        # Get existing chromosomes
        existing = {ind['chromosome'] for ind in self.population}

        # Sample new individuals from pool
        available = [ch for ch in self.pool if ch not in existing]
        n_new = min(target_size - current_size, len(available))

        if n_new > 0:
            new_chromosomes = random.sample(available, n_new)
            for ch in new_chromosomes:
                self.population.append({'chromosome': ch, 'fitness': None})

        logging.info(f"Regenerated {n_new} new individuals (pool: {len(available)} available)")
        return n_new
