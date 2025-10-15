#!/usr/bin/env python
"""
Experiment configuration for ParticleNet methodology comparison.

Defines signal points, background categories, and training parameters
for comparing multi-classification vs binary-classification approaches.
"""

from typing import Dict, List, Tuple
import logging


class ExperimentConfig:
    """
    Configuration for ParticleNet methodology comparison experiment.

    Defines the 3 signal points and 3 background categories for testing
    multi-classification vs binary-classification performance.
    """

    # Signal points for the experiment: (MHc, MA) pairs
    SIGNAL_POINTS = [
        ("MHc160", "MA85"),   # (160, 85)
        ("MHc130", "MA90"),   # (130, 90)
        ("MHc100", "MA95"),   # (100, 95)
        #("MHc115", "MA87"),   # (115, 87)
        #("MHc145", "MA92"),   # (145, 92)
        #("MHc160", "MA98")    # (160, 98)
    ]

    # Background categories and their corresponding sample names
    BACKGROUND_CATEGORIES = {
        "nonprompt": "TTLL_powheg",
        "diboson": "WZTo3LNu_amcatnlo",
        "ttZ": "TTZToLLNuNu"
    }

    # Sample prefixes
    SIGNAL_PREFIX = "TTToHcToWAToMuMu-"
    BACKGROUND_PREFIX = "Skim_TriLep_"

    # Training configuration
    FOLD = 3  # Use only fold 3 for this test

    # Valid channel options
    VALID_CHANNELS = ["Run1E2Mu", "Run3Mu"]

    def __init__(self, channel="Run3Mu"):
        """
        Initialize experiment configuration.

        Args:
            channel: Channel for training (Run1E2Mu or Run3Mu)
        """
        if channel not in self.VALID_CHANNELS:
            raise ValueError(f"Invalid channel '{channel}'. Must be one of {self.VALID_CHANNELS}")
        self.channel = channel

    # Model parameters (can be overridden)
    DEFAULT_MODEL_PARAMS = {
        "model": "ParticleNet",
        "nNodes": 128,
        "max_epochs": 81,
        "optimizer": "Adam",
        "initLR": 0.001,
        "weight_decay": 1e-4,
        "scheduler": "StepLR",
        "loss_type": "weighted_ce",
        "dropout_p": 0.25
    }

    @classmethod
    def get_signal_samples(cls) -> List[str]:
        """Get list of full signal sample names."""
        signal_samples = []
        for mhc, ma in cls.SIGNAL_POINTS:
            sample_name = f"{cls.SIGNAL_PREFIX}{mhc}_{ma}"
            signal_samples.append(sample_name)
        return signal_samples

    @classmethod
    def get_background_samples(cls) -> List[str]:
        """Get list of full background sample names."""
        background_samples = []
        for bg_name in cls.BACKGROUND_CATEGORIES.values():
            sample_name = f"{cls.BACKGROUND_PREFIX}{bg_name}"
            background_samples.append(sample_name)
        return background_samples

    @classmethod
    def get_background_sample_by_category(cls, category: str) -> str:
        """Get full background sample name for a specific category."""
        if category not in cls.BACKGROUND_CATEGORIES:
            raise ValueError(f"Unknown background category: {category}")
        bg_name = cls.BACKGROUND_CATEGORIES[category]
        return f"{cls.BACKGROUND_PREFIX}{bg_name}"

    @classmethod
    def get_multiclass_training_scenarios(cls) -> List[Dict]:
        """Get all multi-class training scenarios."""
        scenarios = []
        signal_samples = cls.get_signal_samples()
        background_samples = cls.get_background_samples()

        for signal in signal_samples:
            scenario = {
                "type": "multiclass",
                "signal": signal,
                "backgrounds": background_samples,
                "num_classes": 4,  # signal + 3 backgrounds
                "description": f"{signal} vs. (nonprompt + diboson + ttZ)"
            }
            scenarios.append(scenario)

        return scenarios

    @classmethod
    def get_binary_training_scenarios(cls) -> List[Dict]:
        """Get all binary training scenarios."""
        scenarios = []
        signal_samples = cls.get_signal_samples()

        for signal in signal_samples:
            for bg_category, bg_sample_base in cls.BACKGROUND_CATEGORIES.items():
                bg_sample = f"{cls.BACKGROUND_PREFIX}{bg_sample_base}"
                scenario = {
                    "type": "binary",
                    "signal": signal,
                    "background": bg_sample,
                    "background_category": bg_category,
                    "num_classes": 2,  # signal vs background
                    "description": f"{signal} vs. {bg_category}"
                }
                scenarios.append(scenario)

        return scenarios

    @classmethod
    def get_all_training_scenarios(cls) -> List[Dict]:
        """Get both multi-class and binary training scenarios."""
        scenarios = []
        scenarios.extend(cls.get_multiclass_training_scenarios())
        scenarios.extend(cls.get_binary_training_scenarios())
        return scenarios

    def log_experiment_summary(self) -> None:
        """Log comprehensive experiment configuration."""
        logging.info("=" * 60)
        logging.info("METHODOLOGY COMPARISON EXPERIMENT CONFIGURATION")
        logging.info("=" * 60)

        # Signal points
        logging.info("Signal Points:")
        for i, (mhc, ma) in enumerate(self.SIGNAL_POINTS, 1):
            signal_name = f"{self.SIGNAL_PREFIX}{mhc}_{ma}"
            logging.info(f"  {i}. {signal_name} ({mhc.replace('MHc', '')}, {ma.replace('MA', '')})")

        # Background categories
        logging.info("\nBackground Categories:")
        for i, (category, sample) in enumerate(self.BACKGROUND_CATEGORIES.items(), 1):
            full_name = f"{self.BACKGROUND_PREFIX}{sample}"
            logging.info(f"  {i}. {category}: {full_name}")

        # Training configuration
        logging.info(f"\nTraining Configuration:")
        logging.info(f"  Fold: {self.FOLD} (single fold test)")
        logging.info(f"  Channel: {self.channel}")

        # Model parameters
        logging.info(f"\nDefault Model Parameters:")
        for param, value in self.DEFAULT_MODEL_PARAMS.items():
            logging.info(f"  {param}: {value}")

        # Training matrix
        multiclass_scenarios = self.get_multiclass_training_scenarios()
        binary_scenarios = self.get_binary_training_scenarios()

        logging.info(f"\nTraining Matrix:")
        logging.info(f"  Multi-class scenarios: {len(multiclass_scenarios)}")
        logging.info(f"  Binary scenarios: {len(binary_scenarios)}")
        logging.info(f"  Total training jobs: {len(multiclass_scenarios) + len(binary_scenarios)}")

        logging.info("\nMulti-Class Training:")
        for scenario in multiclass_scenarios:
            logging.info(f"  - {scenario['description']}")

        logging.info("\nBinary Training:")
        for scenario in binary_scenarios:
            logging.info(f"  - {scenario['description']}")

        logging.info("=" * 60)


def get_experiment_config(channel="Run3Mu") -> ExperimentConfig:
    """
    Factory function to get experiment configuration.

    Args:
        channel: Channel for training (Run1E2Mu or Run3Mu)

    Returns:
        Initialized ExperimentConfig instance
    """
    return ExperimentConfig(channel=channel)


if __name__ == "__main__":
    # Test the configuration
    import logging
    logging.basicConfig(level=logging.INFO)

    config = get_experiment_config()
    config.log_experiment_summary()

    print("\nSignal samples:", config.get_signal_samples())
    print("Background samples:", config.get_background_samples())
    print(f"Total scenarios: {len(config.get_all_training_scenarios())}")
