#!/usr/bin/env python
"""Configuration loader for Genetic Algorithm optimization."""

import json
import os
import numpy as np
from scipy.stats import loguniform
from typing import Dict, List, Any


class GAConfigLoader:
    """Loads and processes GA configuration from JSON file."""

    def __init__(self, config_path: str = None):
        """Initialize config loader with optional custom path."""
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, '..', 'configs', 'GAConfig.json')

        self.config_path = config_path
        self.config = self._load_config()
        self._hyperparameter_spaces = None
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"GA configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _get_config_section(self, section_name: str, defaults: Dict = None) -> Dict[str, Any]:
        """Get config section with comments filtered out."""
        section = self.config.get(section_name, defaults or {})
        return {k: v for k, v in section.items() if k != 'comment'}

    def get_ga_parameters(self) -> Dict[str, Any]:
        """Get GA algorithm parameters."""
        return self._get_config_section('ga_parameters')

    def get_training_parameters(self) -> Dict[str, Any]:
        """Get training parameters."""
        return self._get_config_section('training_parameters')

    def get_background_groups(self) -> Dict[str, List[str]]:
        """Get background group definitions."""
        return self._get_config_section('background_groups')

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self._get_config_section('dataset_config')

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._get_config_section('model_config')

    def get_execution_config(self) -> Dict[str, Any]:
        """Get execution configuration."""
        return self._get_config_section('execution_config')

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self._get_config_section('output_config')

    def get_overfitting_config(self) -> Dict[str, Any]:
        """Get overfitting detection configuration."""
        defaults = {'enabled': False, 'p_value_threshold': 0.05, 'test_folds': [3]}
        return self._get_config_section('overfitting_detection', defaults)

    def generate_hyperparameter_spaces(self) -> List[List[Any]]:
        """Generate hyperparameter search spaces from configuration."""
        if self._hyperparameter_spaces is not None:
            return self._hyperparameter_spaces

        search_space = self.config['hyperparameter_search_space']
        spaces = []

        # Order: nNodes, optimizer, initLR, weight_decay, scheduler
        spaces.append(search_space['nNodes']['values'])
        spaces.append(search_space['optimizers']['values'])

        # Log-uniform sampled parameters
        for param_name in ['initLR', 'weight_decay']:
            param_config = search_space[param_name]
            values = [
                round(value, param_config['round_digits'])
                for value in loguniform.rvs(
                    param_config['min'],
                    param_config['max'],
                    size=param_config['samples']
                )
            ]
            spaces.append(values)

        spaces.append(search_space['schedulers']['values'])

        self._hyperparameter_spaces = spaces
        return spaces

    def get_mutation_thresholds(self) -> List[float]:
        """Get mutation thresholds for GA evolution."""
        return self.config['ga_parameters']['mutation_thresholds']

    def get_population_size(self) -> int:
        """Get population size for GA."""
        return self.config['ga_parameters']['population_size']

    def get_max_iterations(self) -> int:
        """Get maximum number of GA iterations."""
        return self.config['ga_parameters']['max_iterations']

    def get_evolution_ratio(self) -> float:
        """Get evolution ratio (fraction of population to replace)."""
        return self.config['ga_parameters']['evolution_ratio']

    def get_fitness_metric(self) -> str:
        """Get fitness metric name."""
        return self.config['ga_parameters']['fitness_metric']

    def _validate_config(self) -> None:
        """Validate configuration for consistency and correctness."""
        self._validate_mutation_thresholds()
        self._validate_ga_parameters()
        self._validate_training_parameters()
        self._validate_hyperparameter_space()
        self._validate_overfitting_config()

    def _validate_type(self, value, expected_types, param_name: str) -> None:
        """Validate parameter type."""
        if not isinstance(value, expected_types):
            type_names = expected_types.__name__ if hasattr(expected_types, '__name__') else str(expected_types)
            raise ValueError(f"{param_name} must be {type_names}, got {type(value).__name__}")

    def _validate_range(self, value, min_val, max_val, param_name: str, inclusive: str = 'both') -> None:
        """Validate parameter is within range."""
        if inclusive == 'both':
            valid = min_val <= value <= max_val
            range_str = f"[{min_val}, {max_val}]"
        elif inclusive == 'neither':
            valid = min_val < value < max_val
            range_str = f"({min_val}, {max_val})"
        elif inclusive == 'left':
            valid = min_val <= value < max_val
            range_str = f"[{min_val}, {max_val})"
        else:  # right
            valid = min_val < value <= max_val
            range_str = f"({min_val}, {max_val}]"

        if not valid:
            raise ValueError(f"{param_name} must be in range {range_str}, got {value}")

    def _validate_positive_int(self, value, param_name: str) -> None:
        """Validate parameter is a positive integer."""
        self._validate_type(value, int, param_name)
        if value <= 0:
            raise ValueError(f"{param_name} must be positive, got {value}")

    def _validate_mutation_thresholds(self) -> None:
        """Validate mutation thresholds match hyperparameter count and are in valid range."""
        thresholds = self.config['ga_parameters'].get('mutation_thresholds', [])
        expected_count = 5  # nNodes, optimizer, initLR, weight_decay, scheduler

        if len(thresholds) != expected_count:
            raise ValueError(
                f"mutation_thresholds must have exactly {expected_count} values "
                f"(nNodes, optimizer, initLR, weight_decay, scheduler). Got {len(thresholds)}: {thresholds}"
            )

        for i, threshold in enumerate(thresholds):
            self._validate_type(threshold, (int, float), f"mutation_thresholds[{i}]")
            self._validate_range(threshold, 0, 1, f"mutation_thresholds[{i}]")

    def _validate_ga_parameters(self) -> None:
        """Validate GA algorithm parameters."""
        ga_params = self.config['ga_parameters']

        self._validate_positive_int(ga_params.get('population_size'), 'population_size')
        self._validate_positive_int(ga_params.get('max_iterations'), 'max_iterations')

        ratio = ga_params.get('evolution_ratio')
        self._validate_type(ratio, (int, float), 'evolution_ratio')
        self._validate_range(ratio, 0, 1, 'evolution_ratio', inclusive='right')

        metric = ga_params.get('fitness_metric', '')
        if '/' not in metric:
            raise ValueError(f"fitness_metric must be 'metric_type/split_type' (e.g., 'loss/valid'), got '{metric}'")

    def _validate_training_parameters(self) -> None:
        """Validate training parameters."""
        train_params = self.config['training_parameters']

        self._validate_positive_int(train_params.get('max_epochs'), 'max_epochs')
        self._validate_positive_int(train_params.get('batch_size'), 'batch_size')
        self._validate_positive_int(train_params.get('early_stopping_patience'), 'early_stopping_patience')

        dropout = train_params.get('dropout_p')
        self._validate_type(dropout, (int, float), 'dropout_p')
        self._validate_range(dropout, 0, 1, 'dropout_p')

        for fold_key in ['train_folds', 'valid_folds']:
            folds = train_params.get(fold_key, [])
            self._validate_type(folds, list, fold_key)
            for i, fold in enumerate(folds):
                self._validate_type(fold, int, f"{fold_key}[{i}]")
                if fold < 0:
                    raise ValueError(f"{fold_key}[{i}] must be non-negative, got {fold}")

    def _validate_hyperparameter_space(self) -> None:
        """Validate hyperparameter search space definitions."""
        search_space = self.config.get('hyperparameter_search_space', {})

        valid_optimizers = ['RMSprop', 'Adam', 'Adadelta']
        for opt in search_space.get('optimizers', {}).get('values', []):
            if opt not in valid_optimizers:
                raise ValueError(f"Invalid optimizer '{opt}'. Valid: {valid_optimizers}")

        valid_schedulers = ['StepLR', 'ExponentialLR', 'CyclicLR', 'ReduceLROnPlateau']
        for sched in search_space.get('schedulers', {}).get('values', []):
            if sched not in valid_schedulers:
                raise ValueError(f"Invalid scheduler '{sched}'. Valid: {valid_schedulers}")

        for n in search_space.get('nNodes', {}).get('values', []):
            self._validate_positive_int(n, 'nNodes value')

        for param_name in ['initLR', 'weight_decay']:
            param_config = search_space.get(param_name, {})
            if param_config.get('type') == 'log_uniform':
                min_val = param_config.get('min')
                max_val = param_config.get('max')
                samples = param_config.get('samples')

                self._validate_type(min_val, (int, float), f"{param_name} min")
                self._validate_type(max_val, (int, float), f"{param_name} max")
                if min_val <= 0:
                    raise ValueError(f"{param_name} min must be positive, got {min_val}")
                if max_val <= 0:
                    raise ValueError(f"{param_name} max must be positive, got {max_val}")
                if min_val >= max_val:
                    raise ValueError(f"{param_name} min ({min_val}) must be < max ({max_val})")

                self._validate_positive_int(samples, f"{param_name} samples")

    def _validate_overfitting_config(self) -> None:
        """Validate overfitting detection configuration."""
        overfitting_config = self.config.get('overfitting_detection', {})

        if 'enabled' in overfitting_config:
            enabled = overfitting_config['enabled']
            self._validate_type(enabled, bool, 'overfitting_detection.enabled')

            if enabled:
                p_threshold = overfitting_config.get('p_value_threshold')
                self._validate_type(p_threshold, (int, float), 'overfitting_detection.p_value_threshold')
                self._validate_range(p_threshold, 0, 1, 'overfitting_detection.p_value_threshold', inclusive='neither')

                test_folds = overfitting_config.get('test_folds', [])
                self._validate_type(test_folds, list, 'overfitting_detection.test_folds')
                for i, fold in enumerate(test_folds):
                    self._validate_type(fold, int, f'overfitting_detection.test_folds[{i}]')
                    if fold < 0:
                        raise ValueError(f'overfitting_detection.test_folds[{i}] must be non-negative, got {fold}')

    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return self.config

    def summary(self) -> str:
        """Generate configuration summary string."""
        lines = ["=" * 60, "GA CONFIGURATION SUMMARY", "=" * 60]

        ga_params = self.get_ga_parameters()
        lines.extend([
            f"Population size: {ga_params['population_size']}",
            f"Max iterations: {ga_params['max_iterations']}",
            f"Evolution ratio: {ga_params['evolution_ratio']}",
            f"Fitness metric: {ga_params['fitness_metric']}"
        ])

        lines.append("\nHyperparameter Search Space:")
        search_space = self.config['hyperparameter_search_space']
        for param_name, param_config in search_space.items():
            if param_config.get('type') == 'discrete':
                lines.append(f"  {param_name}: {param_config['values']}")
            elif param_config.get('type') == 'log_uniform':
                lines.append(f"  {param_name}: log-uniform [{param_config['min']}, {param_config['max']}]")
                lines.append(f"    ({param_config['samples']} samples, {param_config['round_digits']} digits)")

        lines.append("\nBackground Groups:")
        for group_name, samples in self.get_background_groups().items():
            lines.append(f"  {group_name}: {samples}")

        train_params = self.get_training_parameters()
        lines.extend([
            "\nTraining Parameters:",
            f"  Max epochs: {train_params['max_epochs']}",
            f"  Batch size: {train_params['batch_size']}",
            f"  Loss type: {train_params['loss_type']}",
            f"  Train folds: {train_params['train_folds']}",
            f"  Valid folds: {train_params['valid_folds']}",
            "=" * 60
        ])

        return "\n".join(lines)


def load_ga_config(config_path: str = None) -> GAConfigLoader:
    """Factory function to load GA configuration."""
    return GAConfigLoader(config_path)


if __name__ == "__main__":
    config = load_ga_config()
    print(config.summary())

    print("\nGenerated Hyperparameter Spaces:")
    spaces = config.generate_hyperparameter_spaces()
    print(f"  nNodes: {len(spaces[0])} values - {spaces[0]}")
    print(f"  optimizers: {len(spaces[1])} values - {spaces[1]}")
    print(f"  initLR: {len(spaces[2])} values (showing first 5) - {spaces[2][:5]}")
    print(f"  weight_decay: {len(spaces[3])} values (showing first 5) - {spaces[3][:5]}")
    print(f"  schedulers: {len(spaces[4])} values - {spaces[4]}")
    print("\nConfiguration loaded successfully!")
