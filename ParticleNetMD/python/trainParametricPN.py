#!/usr/bin/env python3
"""Train fixed-mHc parametric ParticleNetMD models: P(class | event, mA)."""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from SglConfig import load_sgl_config
from ParametricDataPipeline import create_parametric_data_pipeline
from TrainingOrchestrator import create_training_orchestrator
from ResultPersistence import create_result_persistence


def parse_ma_values(value: str) -> List[int]:
    try:
        return [int(item.strip()) for item in value.split(',') if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid mA list '{value}'") from exc


class ParametricConfig:
    """Config adapter compatible with existing training modules."""

    def __init__(self, mhc: int, ma_values: List[int], channel: str,
                 config_path: str = None):
        if config_path is None:
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'configs',
                'ParametricPNConfig.json',
            )
        self.sgl_config = load_sgl_config(config_path)

        train_params = self.sgl_config.get_training_parameters()
        model_config = self.sgl_config.get_model_config()
        optim_config = self.sgl_config.get_optimization_config()
        bg_config = self.sgl_config.get_background_config()
        dataset_config = self.sgl_config.get_dataset_config()
        system_config = self.sgl_config.get_system_config()
        output_config = self.sgl_config.get_output_config()
        param_config = self.sgl_config.config.get('parametric_config', {})

        if not ma_values:
            ma_values = [int(v) for v in param_config.get('ma_values', [85, 90, 95])]
        ma_values = sorted(int(v) for v in ma_values)

        self.args = argparse.Namespace()
        self.args.parametric = True
        self.args.mhc = int(mhc)
        self.args.ma_values = ma_values
        self.args.ma_center = float(param_config.get('ma_center', 90.0))
        self.args.ma_scale = float(param_config.get('ma_scale', 10.0))
        self.args.signal_mass_balance = param_config.get('signal_mass_balance', 'event_count')
        self.args.background_hypotheses = param_config.get('background_hypotheses', 'duplicate_all')
        self.args.pilot_max_train_events_per_fold_per_class = int(
            param_config.get('pilot_max_train_events_per_fold_per_class', 1000)
        )
        self.args.pilot_max_eval_events_per_fold_per_class = int(
            param_config.get('pilot_max_eval_events_per_fold_per_class', 300)
        )
        self.args.data_loading_workers = int(param_config.get('data_loading_workers', 4))
        self.args.num_graph_features = 9

        self.args.signal = self._parametric_signal_name()
        self.args.channel = channel
        self.args.train_folds = train_params['train_folds']
        self.args.valid_folds = train_params['valid_folds']
        self.args.test_folds = train_params['test_folds']
        self.args.fold = train_params['test_folds'][0] if train_params['test_folds'] else 4
        self.args.max_epochs = train_params['max_epochs']
        self.args.batch_size = train_params['batch_size']
        self.args.dropout_p = train_params['dropout_p']
        self.args.early_stopping_patience = train_params['early_stopping_patience']
        self.args.loss_type = train_params.get('loss_type', 'disco')
        self.args.balance = train_params['balance_weights']
        self.args.max_events_per_fold_per_class = train_params.get('max_events_per_fold_per_class')
        self.args.augment_phi_rotation = train_params.get('augment_phi_rotation', True)

        disco_params = self.sgl_config.config.get('disco_parameters', {})
        self.args.disco_lambda = disco_params.get('disco_lambda', 0.1)

        self.args.model = model_config['default_model']
        self.args.nNodes = model_config['nNodes']
        self.args.conv_channels = model_config.get('conv_channels')
        self.args.edge_dropout_p = model_config.get('edge_dropout_p', self.args.dropout_p)

        self.args.optimizer = optim_config['optimizer']
        self.args.initLR = optim_config['initLR']
        self.args.weight_decay = optim_config['weight_decay']
        self.args.scheduler = optim_config['scheduler']

        self.args.signal_prefix = dataset_config['signal_prefix']
        self.args.background_prefix = dataset_config['background_prefix']
        self.args.device = system_config['device']
        self.args.pilot = False
        self.args.debug = system_config['debug']
        self.args.results_dir = output_config.get('results_dir', 'ParametricPN')

        self.use_groups = (bg_config['mode'] == 'groups')
        if self.use_groups:
            self.background_groups = bg_config['background_groups']
            self.backgrounds_list = []
            for samples in self.background_groups.values():
                self.backgrounds_list.extend(samples)
        else:
            self.background_groups = {}
            self.backgrounds_list = bg_config['backgrounds_list']

        self.signal_names = [f"MHc{self.args.mhc}_MA{ma}" for ma in self.args.ma_values]
        self.signal_full_names = [
            self.args.signal_prefix + signal_name for signal_name in self.signal_names
        ]
        self.signal_full_name = self.args.signal_prefix + self.args.signal
        self.background_full_names = [
            self.args.background_prefix + bg for bg in self.backgrounds_list
        ]
        self.num_classes = 1 + len(self.background_groups) if self.use_groups else 1 + len(self.backgrounds_list)

        try:
            self.workdir = os.environ["WORKDIR"]
        except KeyError:
            raise EnvironmentError("WORKDIR environment variable not set. Run 'source setup.sh' first.")

    def _parametric_signal_name(self) -> str:
        ma_label = "_".join(f"MA{ma}" for ma in self.args.ma_values)
        return f"MHc{self.args.mhc}_{ma_label}"

    def get_background_groups_full(self) -> Dict[str, List[str]]:
        if not self.use_groups:
            return {}
        return {
            group_name: [self.args.background_prefix + sample for sample in samples]
            for group_name, samples in self.background_groups.items()
        }

    def get_model_name(self) -> str:
        if self.use_groups:
            bg_identifier = f"{len(self.background_groups)}grp-" + "-".join(self.background_groups.keys())
        else:
            bg_identifier = f"{len(self.backgrounds_list)}bg"

        loss_label = self.args.loss_type
        if loss_label == 'disco':
            lam_str = str(self.args.disco_lambda).replace('.', 'p')
            loss_label = f"discoL{lam_str}"

        edge_label = ""
        if abs(float(self.args.edge_dropout_p) - float(self.args.dropout_p)) > 1e-12:
            edge_str = str(format(float(self.args.edge_dropout_p), ".3f")).rstrip("0").rstrip(".").replace(".", "p")
            edge_label = f"-edgeDrop{edge_str}"

        width_label = f"nNodes{self.args.nNodes}"
        if self.args.conv_channels:
            width_label = "conv" + "x".join(str(int(width)) for width in self.args.conv_channels)

        ma_label = "x".join(str(ma) for ma in self.args.ma_values)
        return (
            f"ParametricPN-MHc{self.args.mhc}-MA{ma_label}-{width_label}-{self.args.optimizer}-"
            f"initLR{str(format(self.args.initLR, '.4f')).replace('.', 'p')}-"
            f"decay{str(format(self.args.weight_decay, '.5f')).replace('.', 'p')}-"
            f"{self.args.scheduler}-{loss_label}{edge_label}-{bg_identifier}"
        )

    def get_output_paths(self, model_name: str) -> Tuple[str, str, str, str]:
        output_path = (
            f"{self.workdir}/ParticleNetMD/{self.args.results_dir}/"
            f"{self.args.channel}/{self.args.signal}/fold-{self.args.fold}"
        )
        if self.args.pilot:
            output_path = (
                f"{self.workdir}/ParticleNetMD/{self.args.results_dir}/"
                f"{self.args.channel}/{self.args.signal}/pilot"
            )

        checkpoint_path = f"{output_path}/models/{model_name}.pt"
        summary_path = f"{output_path}/CSV/{model_name}.csv"
        tree_path = f"{output_path}/trees/{model_name}.root"
        return output_path, checkpoint_path, summary_path, tree_path

    def log_configuration(self) -> None:
        logging.info("=" * 60)
        logging.info("PARAMETRIC PARTICLENETMD CONFIGURATION")
        logging.info("=" * 60)
        logging.info(f"Signal samples: {self.signal_full_names}")
        logging.info(f"mHc: {self.args.mhc}, mA values: {self.args.ma_values}")
        logging.info(f"mA normalization: (mA - {self.args.ma_center}) / {self.args.ma_scale}")
        logging.info(f"Channel: {self.args.channel}")
        logging.info(f"Train folds: {self.args.train_folds}, Valid folds: {self.args.valid_folds}, Test folds: {self.args.test_folds}")
        logging.info(f"Background groups: {self.background_groups}")
        logging.info(
            f"Model: {self.args.model}, graph_features={self.args.num_graph_features}, "
            f"nNodes={self.args.nNodes}, dropout={self.args.dropout_p}, edge_dropout={self.args.edge_dropout_p}"
        )
        logging.info(f"Optimization: {self.args.optimizer}, LR={self.args.initLR}, decay={self.args.weight_decay}")
        logging.info(f"Scheduler: {self.args.scheduler}, Loss: {self.args.loss_type}, DisCo lambda: {self.args.disco_lambda}")
        logging.info(f"Batch size: {self.args.batch_size}, Max epochs: {self.args.max_epochs}")
        logging.info(
            f"Pilot mode: {self.args.pilot}, Balance: {self.args.balance}, "
            f"pilot caps train/eval="
            f"{self.args.pilot_max_train_events_per_fold_per_class}/"
            f"{self.args.pilot_max_eval_events_per_fold_per_class}, "
            f"data loading workers={self.args.data_loading_workers}"
        )
        logging.info("=" * 60)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train parametric ParticleNetMD")
    parser.add_argument("--mhc", type=int, default=130, help="Fixed charged-Higgs mass")
    parser.add_argument("--ma-values", type=parse_ma_values, default=None,
                        help="Comma-separated mA hypotheses, e.g. 85,90,95")
    parser.add_argument("--channel", required=True, choices=["Run1E2Mu", "Run3Mu", "Combined"])
    parser.add_argument("--config", default=None, help="Config JSON path")
    parser.add_argument("--device", default=None, help="Override device from config")
    parser.add_argument("--disco-lambda", type=float, default=None, help="Override DisCo lambda")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--max-events-per-fold-per-class", type=int, default=None,
                        help="Override per-fold cap before parametric expansion")
    parser.add_argument("--pilot-max-train-events-per-fold-per-class", type=int, default=None,
                        help="Override pilot training cap per mass/class/fold before parametric expansion")
    parser.add_argument("--pilot-max-eval-events-per-fold-per-class", type=int, default=None,
                        help="Override pilot valid/test cap per mass/class/fold before parametric expansion")
    parser.add_argument("--data-loading-workers", type=int, default=None,
                        help="Number of threads for ParametricPN dataset file loading")
    parser.add_argument("--pilot", action="store_true", help="Run reduced pilot training")
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = ParametricConfig(args.mhc, args.ma_values, args.channel, args.config)

    if args.device is not None:
        config.args.device = args.device
    if args.disco_lambda is not None:
        config.args.disco_lambda = args.disco_lambda
    if args.max_epochs is not None:
        config.args.max_epochs = args.max_epochs
    if args.max_events_per_fold_per_class is not None:
        config.args.max_events_per_fold_per_class = args.max_events_per_fold_per_class
    if args.pilot_max_train_events_per_fold_per_class is not None:
        config.args.pilot_max_train_events_per_fold_per_class = args.pilot_max_train_events_per_fold_per_class
    if args.pilot_max_eval_events_per_fold_per_class is not None:
        config.args.pilot_max_eval_events_per_fold_per_class = args.pilot_max_eval_events_per_fold_per_class
    if args.data_loading_workers is not None:
        config.args.data_loading_workers = args.data_loading_workers
    config.args.pilot = args.pilot

    logging.basicConfig(
        level=logging.DEBUG if config.args.debug else logging.INFO,
        format='%(levelname)s - %(message)s'
    )
    config.log_configuration()

    data_pipeline = create_parametric_data_pipeline(config)
    data_pipeline.create_datasets()
    data_pipeline.log_dataset_info()
    data_pipeline.create_data_loaders(batch_size=config.args.batch_size)
    data_pipeline.validate_data_integrity()

    batch_info = data_pipeline.get_sample_batch_info()
    logging.info(
        f"Sample batch: {batch_info['batch_size']} events, "
        f"{batch_info['num_nodes']} nodes, graph_features={batch_info['graph_features']}"
    )
    if batch_info['graph_features'] != config.args.num_graph_features:
        raise RuntimeError(
            f"Expected {config.args.num_graph_features} graph features, got {batch_info['graph_features']}"
        )

    orchestrator = create_training_orchestrator(config, data_pipeline)
    model_name = config.get_model_name()
    output_paths = config.get_output_paths(model_name)
    output_path, checkpoint_path, summary_path, tree_path = output_paths

    persistence = create_result_persistence(config)
    persistence.create_output_directories(output_paths)
    persistence.log_output_paths(output_paths, model_name)

    orchestrator.setup_training_infrastructure(model_name, checkpoint_path)
    training_results = orchestrator.train()
    test_results = orchestrator.evaluate_final_performance()
    training_results.update(test_results)

    orchestrator.save_training_summary(summary_path)
    persistence.save_predictions_to_root(orchestrator.get_model(), data_pipeline, orchestrator.device, tree_path)
    persistence.save_performance_summary(training_results, model_name, output_path)
    persistence.save_model_info(orchestrator.get_model(), model_name, output_path)
    persistence.save_ga_compatible_json(training_results, model_name, output_path)

    logging.info("=" * 60)
    logging.info("PARAMETRIC TRAINING COMPLETED")
    logging.info(f"Final test accuracy: {training_results['test_accuracy'] * 100:.2f}%")
    logging.info(f"Results saved to: {output_path}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
