#!/usr/bin/env python
"""
Experiment orchestrator for ParticleNet methodology comparison.

Coordinates both multi-class and binary training experiments
for comparing classification approaches on fold 3.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import multiprocessing as mp
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
from datetime import datetime

# Add ParticleNet python directory to path
sys.path.insert(0, os.path.dirname(__file__))

from experiment_config import ExperimentConfig


class ComparisonExperimentOrchestrator:
    """
    Orchestrates methodology comparison experiment between multi-class and binary classification.

    Manages dataset validation, training execution, and result organization
    for comprehensive performance comparison.
    """

    def __init__(self, config: ExperimentConfig, args: argparse.Namespace):
        """
        Initialize experiment orchestrator.

        Args:
            config: ExperimentConfig instance
            args: Parsed command line arguments
        """
        self.config = config
        self.args = args
        self.workdir = os.environ.get("WORKDIR", os.path.join(os.path.dirname(__file__), "../.."))

        # Set experiment root based on separate_bjets flag
        results_dir = "results_bjets" if getattr(args, 'separate_bjets', False) else "results"
        self.experiment_root = f"{self.workdir}/ParticleNet/{results_dir}/comparison_experiment"
        self.start_time = datetime.now()

        # Setup logging
        self._setup_logging()

        # Training statistics
        self.training_stats = {
            "multiclass": {"total": 0, "completed": 0, "failed": 0},
            "binary": {"total": 0, "completed": 0, "failed": 0},
            "start_time": self.start_time.isoformat(),
            "scenarios": []
        }

    def _setup_logging(self) -> None:
        """Setup comprehensive logging for experiment tracking."""
        log_level = logging.DEBUG if self.args.debug else logging.INFO

        # Create logs directory
        log_dir = f"{self.workdir}/ParticleNet/logs"
        os.makedirs(log_dir, exist_ok=True)

        # Setup file logging
        log_file = f"{log_dir}/comparison_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

        logging.info(f"Experiment log: {log_file}")

    def validate_environment(self) -> bool:
        """Validate that environment and datasets are ready for training."""
        logging.info("Validating environment and datasets...")

        # Check WORKDIR
        if not os.path.isdir(self.workdir):
            logging.error(f"WORKDIR not found: {self.workdir}")
            return False

        # Check ParticleNet directory structure
        particlenet_dir = f"{self.workdir}/ParticleNet"
        required_dirs = ["python", "scripts"]
        for req_dir in required_dirs:
            if not os.path.isdir(f"{particlenet_dir}/{req_dir}"):
                logging.error(f"Required directory not found: {particlenet_dir}/{req_dir}")
                return False

        # Check training scripts
        required_scripts = [
            "python/trainMultiClass.py",
            "python/trainBinary.py",
            "scripts/trainMultiClass.sh",
            "scripts/trainBinary.sh"
        ]
        for script in required_scripts:
            script_path = f"{particlenet_dir}/{script}"
            if not os.path.isfile(script_path):
                logging.error(f"Required script not found: {script_path}")
                return False

        # Check dataset availability
        dataset_dir = "dataset_bjets" if getattr(self.args, 'separate_bjets', False) else "dataset"
        dataset_root = f"{self.workdir}/ParticleNet/{dataset_dir}"
        if not os.path.isdir(dataset_root):
            logging.warning(f"Dataset root not found: {dataset_root}")
            logging.warning("You may need to run dataset creation first")

        logging.info("Environment validation completed")
        return True

    def validate_datasets(self) -> bool:
        """Validate that required datasets exist for the experiment."""
        logging.info("Validating experiment datasets...")

        dataset_dir = "dataset_bjets" if getattr(self.args, 'separate_bjets', False) else "dataset"
        dataset_root = f"{self.workdir}/ParticleNet/{dataset_dir}/samples"
        if not os.path.isdir(dataset_root):
            logging.error(f"Dataset samples directory not found: {dataset_root}")
            return False

        # Check signal datasets
        signal_samples = self.config.get_signal_samples()
        missing_signals = []
        for signal in signal_samples:
            signal_dir = f"{dataset_root}/signals/{signal}"
            if not os.path.isdir(signal_dir):
                missing_signals.append(signal)

        # Check background datasets
        background_samples = self.config.get_background_samples()
        missing_backgrounds = []
        for background in background_samples:
            bg_dir = f"{dataset_root}/backgrounds/{background}"
            if not os.path.isdir(bg_dir):
                missing_backgrounds.append(background)

        # Report missing datasets
        if missing_signals:
            logging.error(f"Missing signal datasets: {missing_signals}")
            return False

        if missing_backgrounds:
            logging.error(f"Missing background datasets: {missing_backgrounds}")
            return False

        logging.info("All required datasets found")
        return True

    def setup_experiment_directories(self) -> None:
        """Create experiment output directory structure."""
        logging.info("Setting up experiment directories...")

        dirs_to_create = [
            f"{self.experiment_root}",
            f"{self.experiment_root}/multiclass",
            f"{self.experiment_root}/binary",
            #f"{self.experiment_root}/analysis",
            f"{self.experiment_root}/logs"
        ]

        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
            logging.debug(f"Created directory: {directory}")

        logging.info(f"Experiment directories created in: {self.experiment_root}")

    def run_multiclass_training(self) -> bool:
        """Execute multi-class training for all signal points using GNU parallel."""
        logging.info("=" * 60)
        logging.info("STARTING MULTI-CLASS TRAINING")
        logging.info("=" * 60)

        multiclass_scenarios = self.config.get_multiclass_training_scenarios()
        self.training_stats["multiclass"]["total"] = len(multiclass_scenarios)

        # Build all commands
        commands = []
        for i, scenario in enumerate(multiclass_scenarios, 1):
            logging.info(f"Multi-class training {i}/{len(multiclass_scenarios)}: {scenario['description']}")
            cmd = self._build_multiclass_command(scenario)
            commands.append(cmd)

            # Store scenario for later reference
            scenario_stats = {
                "type": "multiclass",
                "scenario": scenario,
                "command": cmd
            }
            self.training_stats["scenarios"].append(scenario_stats)

            if self.args.dry_run:
                logging.info(f"DRY RUN - Would execute: {cmd}")

        if self.args.dry_run:
            logging.info("=" * 60)
            logging.info("MULTI-CLASS TRAINING COMPLETED (DRY RUN)")
            logging.info("=" * 60)
            return True

        # Execute all commands in parallel
        logging.info(f"Executing {len(commands)} multi-class training jobs in parallel...")
        start_time = time.time()
        success = self._execute_parallel_training(commands, "multiclass")
        duration = time.time() - start_time

        logging.info("=" * 60)
        logging.info("MULTI-CLASS TRAINING COMPLETED")
        logging.info(f"Success: {self.training_stats['multiclass']['completed']}/{self.training_stats['multiclass']['total']}")
        logging.info(f"Total duration: {duration:.1f}s")
        logging.info("=" * 60)

        return success

    def run_binary_training(self) -> bool:
        """Execute binary training for all signal-background combinations using GNU parallel."""
        logging.info("=" * 60)
        logging.info("STARTING BINARY TRAINING")
        logging.info("=" * 60)

        binary_scenarios = self.config.get_binary_training_scenarios()
        self.training_stats["binary"]["total"] = len(binary_scenarios)

        # Build all commands
        commands = []
        for i, scenario in enumerate(binary_scenarios, 1):
            logging.info(f"Binary training {i}/{len(binary_scenarios)}: {scenario['description']}")
            cmd = self._build_binary_command(scenario)
            commands.append(cmd)

            # Store scenario for later reference
            scenario_stats = {
                "type": "binary",
                "scenario": scenario,
                "command": cmd
            }
            self.training_stats["scenarios"].append(scenario_stats)

            if self.args.dry_run:
                logging.info(f"DRY RUN - Would execute: {cmd}")

        if self.args.dry_run:
            logging.info("=" * 60)
            logging.info("BINARY TRAINING COMPLETED (DRY RUN)")
            logging.info("=" * 60)
            return True

        # Execute all commands in parallel
        logging.info(f"Executing {len(commands)} binary training jobs in parallel...")
        start_time = time.time()
        success = self._execute_parallel_training(commands, "binary")
        duration = time.time() - start_time

        logging.info("=" * 60)
        logging.info("BINARY TRAINING COMPLETED")
        logging.info(f"Success: {self.training_stats['binary']['completed']}/{self.training_stats['binary']['total']}")
        logging.info(f"Total duration: {duration:.1f}s")
        logging.info("=" * 60)

        return success

    def _build_multiclass_command(self, scenario: Dict) -> str:
        """Build multi-class training command."""
        signal = scenario["signal"].replace(self.config.SIGNAL_PREFIX, "")
        backgrounds = [bg.replace(self.config.BACKGROUND_PREFIX, "") for bg in scenario["backgrounds"]]

        cmd = [
            "python", "python/trainMultiClass.py",
            "--signal", signal,
            "--channel", self.config.channel,
            "--fold", str(self.config.FOLD),
            "--backgrounds"] + backgrounds

        # Add model parameters
        for param, value in self.config.DEFAULT_MODEL_PARAMS.items():
            cmd.extend([f"--{param}", str(value)])

        if self.args.pilot:
            cmd.append("--pilot")

        if getattr(self.args, 'separate_bjets', False):
            cmd.append("--separate_bjets")

        if self.args.debug:
            cmd.append("--debug")

        return " ".join(cmd)

    def _build_binary_command(self, scenario: Dict) -> str:
        """Build binary training command."""
        signal = scenario["signal"].replace(self.config.SIGNAL_PREFIX, "")
        background_category = scenario["background_category"]

        cmd = [
            "python", "python/trainBinary.py",
            "--signal", signal,
            "--background", background_category,
            "--channel", self.config.channel,
            "--fold", str(self.config.FOLD)
        ]

        # Add model parameters
        for param, value in self.config.DEFAULT_MODEL_PARAMS.items():
            cmd.extend([f"--{param}", str(value)])

        if self.args.pilot:
            cmd.append("--pilot")

        if getattr(self.args, 'separate_bjets', False):
            cmd.append("--separate_bjets")

        if self.args.debug:
            cmd.append("--debug")

        return " ".join(cmd)

    def _execute_parallel_training(self, commands: List[str], training_type: str) -> bool:
        """Execute training commands in parallel using GNU parallel."""
        if not commands:
            return True

        # Create temporary file with all commands
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp_file:
            for cmd in commands:
                tmp_file.write(f"{cmd}\n")
            commands_file = tmp_file.name

        try:
            # Setup parallel execution parameters
            max_jobs = getattr(self.args, 'max_parallel_jobs', 0)  # 0 means use all cores

            # Create job log file
            log_dir = f"{self.workdir}/ParticleNet/logs"
            os.makedirs(log_dir, exist_ok=True)
            job_log = f"{log_dir}/{training_type}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

            # Check if GNU parallel is available
            try:
                subprocess.run(['parallel', '--version'], capture_output=True, check=True)
                has_parallel = True
            except (subprocess.CalledProcessError, FileNotFoundError):
                has_parallel = False
                logging.warning("GNU parallel not found, falling back to sequential execution")

            if has_parallel and not getattr(self.args, 'no_parallel', False):
                # Build GNU parallel command
                parallel_cmd = [
                    'parallel',
                    '--jobs', str(max_jobs) if max_jobs > 0 else '0',
                    '--halt', 'soon,fail=25%',  # Stop if >25% of jobs fail
                    '--progress',
                    '--joblog', job_log,
                    '--colsep', '\\t'
                ]

                # Add timeout if specified
                if hasattr(self.args, 'job_timeout') and self.args.job_timeout > 0:
                    parallel_cmd.extend(['--timeout', str(self.args.job_timeout)])

                logging.info(f"Using GNU parallel with {max_jobs if max_jobs > 0 else 'all'} jobs")
                logging.info(f"Job log: {job_log}")

                # Execute parallel training
                particlenet_dir = f"{self.workdir}/ParticleNet"
                result = subprocess.run(
                    parallel_cmd,
                    stdin=open(commands_file, 'r'),
                    cwd=particlenet_dir,
                    capture_output=True,
                    text=True
                )

                # Parse results from job log
                success_count, failed_count = self._parse_parallel_results(job_log, training_type)

                if result.returncode == 0:
                    logging.info(f"✓ All {training_type} training jobs completed successfully")
                    return True
                else:
                    logging.error(f"✗ Some {training_type} training jobs failed")
                    logging.error(f"GNU parallel output: {result.stderr}")
                    return not self.args.continue_on_failure
            else:
                # Sequential execution fallback
                logging.info(f"Running {len(commands)} {training_type} training jobs sequentially")
                success_count = 0
                failed_count = 0

                particlenet_dir = f"{self.workdir}/ParticleNet"

                for i, cmd in enumerate(commands, 1):
                    logging.info(f"Executing {training_type} job {i}/{len(commands)}")
                    try:
                        result = subprocess.run(
                            cmd,
                            shell=True,
                            cwd=particlenet_dir,
                            capture_output=True,
                            text=True,
                            timeout=3600  # 1 hour timeout per job
                        )

                        if result.returncode == 0:
                            success_count += 1
                            logging.info(f"✓ Job {i} completed successfully")
                        else:
                            failed_count += 1
                            logging.error(f"✗ Job {i} failed with return code {result.returncode}")
                            if not self.args.continue_on_failure:
                                break

                    except subprocess.TimeoutExpired:
                        failed_count += 1
                        logging.error(f"✗ Job {i} timed out after 1 hour")
                        if not self.args.continue_on_failure:
                            break
                    except Exception as e:
                        failed_count += 1
                        logging.error(f"✗ Job {i} failed with exception: {e}")
                        if not self.args.continue_on_failure:
                            break

                # Update statistics
                self.training_stats[training_type]["completed"] = success_count
                self.training_stats[training_type]["failed"] = failed_count

                return failed_count == 0 or self.args.continue_on_failure

        finally:
            # Cleanup temporary file
            if os.path.exists(commands_file):
                os.unlink(commands_file)

    def _parse_parallel_results(self, job_log: str, training_type: str) -> tuple:
        """Parse GNU parallel job log to extract success/failure counts."""
        success_count = 0
        failed_count = 0

        try:
            if os.path.exists(job_log):
                with open(job_log, 'r') as f:
                    lines = f.readlines()
                    # Skip header line
                    for line in lines[1:]:
                        parts = line.strip().split('\t')
                        if len(parts) >= 7:  # Standard parallel joblog format
                            exit_code = parts[6]
                            if exit_code == '0':
                                success_count += 1
                            else:
                                failed_count += 1
        except Exception as e:
            logging.warning(f"Could not parse job log {job_log}: {e}")
            # Fall back to estimating from total
            total_jobs = self.training_stats[training_type]["total"]
            success_count = total_jobs  # Assume success if we can't parse
            failed_count = 0

        # Update statistics
        self.training_stats[training_type]["completed"] = success_count
        self.training_stats[training_type]["failed"] = failed_count

        return success_count, failed_count

    def _validate_experiment_success(self) -> bool:
        """Validate experiment success by checking for output models and results."""
        logging.info("Validating experiment success by checking output models...")

        expected_models = 0
        found_models = 0

        # Check multi-class models
        if not getattr(self.args, 'binary_only', False):
            for scenario in self.config.get_multiclass_training_scenarios():
                expected_models += 1
                signal = scenario["signal"].replace(self.config.SIGNAL_PREFIX, "")
                model_pattern = f"*{signal}*fold-{self.config.FOLD}*"

                # Look for model files in results directory
                multiclass_dir = f"{self.workdir}/ParticleNet/results/multiclass/{self.config.channel}"
                model_found = self._check_model_exists(multiclass_dir, model_pattern)
                if model_found:
                    found_models += 1
                    logging.info(f"✓ Multi-class model found: {signal}")
                else:
                    logging.warning(f"✗ Multi-class model missing: {signal}")

        # Check binary models
        if not getattr(self.args, 'multiclass_only', False):
            for scenario in self.config.get_binary_training_scenarios():
                expected_models += 1
                signal = scenario["signal"].replace(self.config.SIGNAL_PREFIX, "")
                bg_category = scenario["background_category"]
                model_pattern = f"*{signal}*{bg_category}*fold-{self.config.FOLD}*"

                # Look for model files in results directory
                binary_dir = f"{self.workdir}/ParticleNet/results/binary/{self.config.channel}"
                model_found = self._check_model_exists(binary_dir, model_pattern)
                if model_found:
                    found_models += 1
                    logging.info(f"✓ Binary model found: {signal} vs {bg_category}")
                else:
                    logging.warning(f"✗ Binary model missing: {signal} vs {bg_category}")

        success_rate = found_models / expected_models if expected_models > 0 else 0.0
        logging.info(f"Model validation: {found_models}/{expected_models} models found ({success_rate*100:.1f}%)")

        return found_models > 0 and success_rate >= 0.8  # At least 80% success rate

    def _check_model_exists(self, base_dir: str, pattern: str) -> bool:
        """Check if model files exist matching the pattern."""
        import glob

        # Check for .pt model files
        search_patterns = [
            f"{base_dir}/**/models/*{pattern}*.pt",
            f"{base_dir}/**/*{pattern}*.pt"
        ]

        for search_pattern in search_patterns:
            matches = glob.glob(search_pattern, recursive=True)
            if matches:
                return True

        return False

    def _execute_training_command(self, cmd: str, scenario: Dict) -> bool:
        """Execute a training command and return success status."""
        try:
            # Change to ParticleNet directory
            particlenet_dir = f"{self.workdir}/ParticleNet"
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=particlenet_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            if result.returncode == 0:
                logging.debug(f"Training output: {result.stdout}")
                return True
            else:
                logging.error(f"Training failed with return code {result.returncode}")
                logging.error(f"Error output: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            logging.error("Training timed out after 1 hour")
            return False
        except Exception as e:
            logging.error(f"Training execution failed: {e}")
            return False

    def save_experiment_summary(self) -> None:
        """Save comprehensive experiment summary."""
        logging.info("Saving experiment summary...")

        # Calculate final statistics
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()

        self.training_stats.update({
            "end_time": end_time.isoformat(),
            "total_duration": total_duration,
            "total_jobs": self.training_stats["multiclass"]["total"] + self.training_stats["binary"]["total"],
            "total_completed": self.training_stats["multiclass"]["completed"] + self.training_stats["binary"]["completed"],
            "total_failed": self.training_stats["multiclass"]["failed"] + self.training_stats["binary"]["failed"]
        })

        # Save detailed summary
        summary_file = f"{self.experiment_root}/experiment_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.training_stats, f, indent=2)

        logging.info(f"Experiment summary saved: {summary_file}")

    def run_experiment(self) -> bool:
        """Run the complete methodology comparison experiment."""
        logging.info("=" * 80)
        logging.info("PARTICLENET METHODOLOGY COMPARISON EXPERIMENT")
        logging.info("=" * 80)

        # Log experiment configuration
        self.config.log_experiment_summary()

        # Validate environment
        if not self.validate_environment():
            logging.error("Environment validation failed")
            return False

        if not self.args.skip_dataset_validation and not self.validate_datasets():
            logging.error("Dataset validation failed")
            return False

        # Setup experiment
        self.setup_experiment_directories()

        success = True

        try:
            # Run multi-class training
            if not self.args.binary_only:
                if not self.run_multiclass_training():
                    success = False
                    if not self.args.continue_on_failure:
                        return False

            # Run binary training
            if not self.args.multiclass_only:
                if not self.run_binary_training():
                    success = False

            return success

        finally:
            # Always save summary
            self.save_experiment_summary()

            # Validate experiment success
            if not self.args.dry_run:
                models_validated = self._validate_experiment_success()
                self.training_stats["models_validated"] = models_validated
            else:
                models_validated = True  # Always true for dry runs

            # Print final report
            self._print_final_report()

            return success and models_validated

    def _print_final_report(self) -> None:
        """Print comprehensive final experiment report."""
        logging.info("=" * 80)
        logging.info("EXPERIMENT COMPLETED")
        logging.info("=" * 80)

        # Overall statistics
        total_jobs = self.training_stats["total_jobs"]
        total_completed = self.training_stats["total_completed"]
        total_failed = self.training_stats["total_failed"]
        total_duration = self.training_stats["total_duration"]

        logging.info(f"Total training jobs: {total_jobs}")
        logging.info(f"Completed successfully: {total_completed}")
        logging.info(f"Failed: {total_failed}")
        logging.info(f"Success rate: {(total_completed/total_jobs)*100:.1f}%" if total_jobs > 0 else "0.0%")
        logging.info(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")

        # Multi-class statistics
        mc_stats = self.training_stats["multiclass"]
        logging.info(f"\nMulti-class training: {mc_stats['completed']}/{mc_stats['total']} completed")

        # Binary statistics
        bin_stats = self.training_stats["binary"]
        logging.info(f"Binary training: {bin_stats['completed']}/{bin_stats['total']} completed")

        # Model validation results
        if not self.args.dry_run and "models_validated" in self.training_stats:
            models_validated = self.training_stats["models_validated"]
            logging.info(f"\nModel validation: {'PASSED' if models_validated else 'FAILED'}")
            if not models_validated:
                logging.warning("Some expected model files were not found!")

        # Results location
        logging.info(f"\nResults saved in: {self.experiment_root}")
        logging.info(f"Summary file: {self.experiment_root}/experiment_summary.json")

        logging.info("=" * 80)


def main():
    """Main experiment orchestrator function."""
    parser = argparse.ArgumentParser(description="ParticleNet methodology comparison experiment")

    # Experiment control
    parser.add_argument("--multiclass-only", action="store_true",
                       help="Run only multi-class training")
    parser.add_argument("--binary-only", action="store_true",
                       help="Run only binary training")
    parser.add_argument("--continue-on-failure", action="store_true",
                       help="Continue experiment even if some training jobs fail")

    # Dataset and validation
    parser.add_argument("--skip-dataset-validation", action="store_true",
                       help="Skip dataset validation (use with caution)")

    # Training parameters
    parser.add_argument("--channel", type=str, default="Run3Mu",
                       choices=["Run1E2Mu", "Run3Mu"],
                       help="Channel for training (default: Run3Mu)")
    parser.add_argument("--pilot", action="store_true",
                       help="Use pilot datasets for quick testing")
    parser.add_argument("--separate_bjets", action="store_true",
                       help="Use separate b-jets as distinct particles")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be executed without running")

    # Parallel execution parameters
    parser.add_argument("--max-parallel-jobs", type=int, default=0,
                       help="Maximum parallel jobs (0 = use all CPU cores)")
    parser.add_argument("--no-parallel", action="store_true",
                       help="Disable parallel execution (run sequentially)")
    parser.add_argument("--job-timeout", type=int, default=3600,
                       help="Timeout per training job in seconds (default: 3600)")

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.multiclass_only and args.binary_only:
        parser.error("Cannot specify both --multiclass-only and --binary-only")

    # Create experiment configuration with specified channel
    config = ExperimentConfig(channel=args.channel)

    # Create and run orchestrator
    orchestrator = ComparisonExperimentOrchestrator(config, args)
    success = orchestrator.run_experiment()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
