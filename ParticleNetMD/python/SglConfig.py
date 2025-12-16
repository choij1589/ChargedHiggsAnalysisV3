#!/usr/bin/env python
"""Configuration loader for single-run ParticleNet training."""

import json
import os
from typing import Dict, List, Any


class SglConfigLoader:
    """Loads and processes single-run training configuration from JSON file."""

    def __init__(self, config_path: str = None):
        """Initialize config loader with optional custom path."""
        if config_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, '..', 'configs', 'SglConfig.json')

        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            return json.load(f)

    def _get_config_section(self, section_name: str, defaults: Dict = None) -> Dict[str, Any]:
        """Get config section with comments filtered out."""
        section = self.config.get(section_name, defaults or {})
        return {k: v for k, v in section.items() if k != 'comment'}

    def get_training_parameters(self) -> Dict[str, Any]:
        """Get training parameters."""
        return self._get_config_section('training_parameters')

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self._get_config_section('model_config')

    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return self._get_config_section('optimization_config')

    def get_background_config(self) -> Dict[str, Any]:
        """Get background configuration."""
        return self._get_config_section('background_config')

    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self._get_config_section('dataset_config')

    def get_system_config(self) -> Dict[str, Any]:
        """Get system configuration."""
        return self._get_config_section('system_config')

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self._get_config_section('output_config')

    def _validate_config(self) -> None:
        """Validate configuration for consistency and correctness."""
        self._validate_training_parameters()
        self._validate_model_config()
        self._validate_optimization_config()
        self._validate_background_config()
        self._validate_dataset_config()
        self._validate_system_config()

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

    def _validate_training_parameters(self) -> None:
        """Validate training parameters."""
        train_params = self.config['training_parameters']

        self._validate_positive_int(train_params.get('max_epochs'), 'max_epochs')
        self._validate_positive_int(train_params.get('batch_size'), 'batch_size')
        self._validate_positive_int(train_params.get('early_stopping_patience'), 'early_stopping_patience')

        dropout = train_params.get('dropout_p')
        self._validate_type(dropout, (int, float), 'dropout_p')
        self._validate_range(dropout, 0, 1, 'dropout_p')

        loss_type = train_params.get('loss_type', '')
        valid_losses = ['weighted_ce', 'sample_normalized', 'focal', 'disco']
        if loss_type not in valid_losses:
            raise ValueError(f"loss_type must be one of {valid_losses}, got '{loss_type}'")

        balance = train_params.get('balance_weights')
        self._validate_type(balance, bool, 'balance_weights')

        for fold_key in ['train_folds', 'valid_folds', 'test_folds']:
            folds = train_params.get(fold_key, [])
            self._validate_type(folds, list, fold_key)
            for i, fold in enumerate(folds):
                self._validate_type(fold, int, f"{fold_key}[{i}]")
                if fold < 0 or fold > 4:
                    raise ValueError(f"{fold_key}[{i}] must be in range [0, 4], got {fold}")

    def _validate_model_config(self) -> None:
        """Validate model configuration."""
        model_config = self.config['model_config']

        model = model_config.get('default_model', '')
        valid_models = ['ParticleNet', 'ParticleNetV2', 'OptimizedParticleNet',
                       'EfficientParticleNet', 'EnhancedParticleNet', 'HierarchicalParticleNet']
        if model not in valid_models:
            raise ValueError(f"default_model must be one of {valid_models}, got '{model}'")

        nNodes = model_config.get('nNodes')
        self._validate_positive_int(nNodes, 'nNodes')

    def _validate_optimization_config(self) -> None:
        """Validate optimization configuration."""
        optim_config = self.config['optimization_config']

        optimizer = optim_config.get('optimizer', '')
        valid_optimizers = ['Adam', 'RMSprop', 'Adadelta']
        if optimizer not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}, got '{optimizer}'")

        scheduler = optim_config.get('scheduler', '')
        valid_schedulers = ['StepLR', 'ExponentialLR', 'CyclicLR', 'ReduceLROnPlateau']
        if scheduler not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}, got '{scheduler}'")

        initLR = optim_config.get('initLR')
        self._validate_type(initLR, (int, float), 'initLR')
        if initLR <= 0:
            raise ValueError(f"initLR must be positive, got {initLR}")

        weight_decay = optim_config.get('weight_decay')
        self._validate_type(weight_decay, (int, float), 'weight_decay')
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

    def _validate_background_config(self) -> None:
        """Validate background configuration."""
        bg_config = self.config['background_config']

        mode = bg_config.get('mode', '')
        if mode not in ['groups', 'individual']:
            raise ValueError(f"background_config.mode must be 'groups' or 'individual', got '{mode}'")

        if mode == 'groups':
            groups = bg_config.get('background_groups', {})
            self._validate_type(groups, dict, 'background_groups')
            if not groups:
                raise ValueError("background_groups cannot be empty when mode='groups'")
            for group_name, samples in groups.items():
                self._validate_type(samples, list, f"background_groups[{group_name}]")
                if not samples:
                    raise ValueError(f"background_groups[{group_name}] cannot be empty")
        else:  # individual
            bg_list = bg_config.get('backgrounds_list', [])
            self._validate_type(bg_list, list, 'backgrounds_list')
            if not bg_list:
                raise ValueError("backgrounds_list cannot be empty when mode='individual'")

    def _validate_dataset_config(self) -> None:
        """Validate dataset configuration."""
        dataset_config = self.config['dataset_config']

        # Note: use_bjets field is deprecated and ignored

        signal_prefix = dataset_config.get('signal_prefix', '')
        if not signal_prefix:
            raise ValueError("signal_prefix cannot be empty")

        background_prefix = dataset_config.get('background_prefix', '')
        if not background_prefix:
            raise ValueError("background_prefix cannot be empty")

    def _validate_system_config(self) -> None:
        """Validate system configuration."""
        system_config = self.config['system_config']

        device = system_config.get('device', '')
        if not (device == 'cpu' or device == 'cuda' or device.startswith('cuda:')):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'cuda:X', got '{device}'")

        pilot = system_config.get('pilot')
        self._validate_type(pilot, bool, 'pilot')

        debug = system_config.get('debug')
        self._validate_type(debug, bool, 'debug')

    def to_dict(self) -> Dict[str, Any]:
        """Return the full configuration as a dictionary."""
        return self.config

    def summary(self) -> str:
        """Generate configuration summary string."""
        lines = ["=" * 60, "TRAINING CONFIGURATION SUMMARY", "=" * 60]

        train_params = self.get_training_parameters()
        lines.extend([
            f"Max epochs: {train_params['max_epochs']}",
            f"Batch size: {train_params['batch_size']}",
            f"Dropout: {train_params['dropout_p']}",
            f"Loss type: {train_params['loss_type']}",
            f"Train folds: {train_params['train_folds']}",
            f"Valid folds: {train_params['valid_folds']}",
            f"Test folds: {train_params['test_folds']}",
            f"Balance weights: {train_params['balance_weights']}"
        ])

        model_config = self.get_model_config()
        lines.extend([
            "\nModel Configuration:",
            f"  Model: {model_config['default_model']}",
            f"  Hidden nodes: {model_config['nNodes']}"
        ])

        optim_config = self.get_optimization_config()
        lines.extend([
            "\nOptimization:",
            f"  Optimizer: {optim_config['optimizer']}",
            f"  Initial LR: {optim_config['initLR']}",
            f"  Weight decay: {optim_config['weight_decay']}",
            f"  Scheduler: {optim_config['scheduler']}"
        ])

        bg_config = self.get_background_config()
        lines.append(f"\nBackground mode: {bg_config['mode']}")
        if bg_config['mode'] == 'groups':
            lines.append("Background Groups:")
            for group_name, samples in bg_config['background_groups'].items():
                lines.append(f"  {group_name}: {samples}")
        else:
            lines.append(f"Background samples: {bg_config['backgrounds_list']}")

        dataset_config = self.get_dataset_config()
        lines.extend([
            "\nDataset Configuration:",
            f"  Use b-jets: {dataset_config['use_bjets']}",
            f"  Signal prefix: {dataset_config['signal_prefix']}",
            f"  Background prefix: {dataset_config['background_prefix']}"
        ])

        system_config = self.get_system_config()
        lines.extend([
            "\nSystem Configuration:",
            f"  Device: {system_config['device']}",
            f"  Pilot mode: {system_config['pilot']}",
            f"  Debug mode: {system_config['debug']}",
            "=" * 60
        ])

        return "\n".join(lines)


def load_sgl_config(config_path: str = None) -> SglConfigLoader:
    """Factory function to load single-run training configuration."""
    return SglConfigLoader(config_path)


if __name__ == "__main__":
    config = load_sgl_config()
    print(config.summary())
    print("\nConfiguration loaded and validated successfully!")
